use anyhow::Result;
use csv::ReaderBuilder;
use rust_bert::Config;
use rust_bert::bert::{BertConfig, BertForSequenceClassification};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use tch::Device;
use tch::nn;
use tch::nn::{VarStore, linear};
//use tch::nn::{OptimizerConfig, VarStore, linear};
//use tch::{Device, Tensor};

#[derive(Debug, Deserialize)]
struct Record {
    label: String,
    message: String,
}

fn encode_label_map(labels: &[String]) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let mut index = 0;
    for label in labels {
        if !map.contains_key(label) {
            map.insert(label.to_string(), index);
            index += 1;
        }
    }
    map
}

fn batchify<T: Clone>(data: &[T], batch_size: usize) -> Vec<Vec<T>> {
    data.chunks(batch_size).map(|c| c.to_vec()).collect()
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut vs = VarStore::new(device);
    // Download and set up model resources
    let config_resource = RemoteResource::new(
        "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
        "bert_config.json",
    );
    let vocab_resource = RemoteResource::new(
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "bert_vocab.txt",
    );
    let weights_resource = RemoteResource::new(
        "https://huggingface.co/bert-base-uncased/resolve/main/rust_model.ot",
        "bert_model.ot",
    );

    let _config_path: std::path::PathBuf = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    // Load dataset CSV
    let path = Path::new("sms_spam_reduced.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

    let mut records = vec![];
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    // Limit records for faster training
    let max_samples = 500;
    records = records.into_iter().take(max_samples).collect();

    let labels = records.iter().map(|r| r.label.clone()).collect::<Vec<_>>();
    let label_map = encode_label_map(&labels);

    let config_path: std::path::PathBuf = config_resource.get_local_path()?;
    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config_json: serde_json::Value = serde_json::from_str(&config_str)?;

    // Tambahkan atau ubah field num_labels
    config_json["architectures"] = serde_json::Value::Array(vec![serde_json::Value::String(
        "BertForSequenceClassification".to_string(),
    )]);
    config_json["num_labels"] =
        serde_json::Value::Number(serde_json::Number::from(label_map.len() as i64));
    println!(
        "Modif config JSON: {}",
        serde_json::to_string_pretty(&config_json)?
    );

    // Simpan ke file baru jika diperlukan atau langsung parse
    let tmp_config_path = std::env::temp_dir().join("bert_config_modified.json");
    std::fs::write(
        &tmp_config_path,
        serde_json::to_string_pretty(&config_json)?,
    )?;

    // Muat konfigurasi dari file baru
    let config_path = "path/bert_config_modified.json";
    let config = BertConfig::from_file(&tmp_config_path);

    // Patch classifier (output layer) dengan jumlah label baru
    let num_labels = 2; // misal 2 kelas (spam/ham)

    // Ambil ukuran hidden dari config
    let hidden_size = config.hidden_size;

    // Buat layer linear baru untuk klasifikasi
    let classifier_layer = nn::linear(
        &vs.root() / "classifier",
        hidden_size,
        num_labels,
        Default::default(),
    );

    // Ganti classifier lama dengan layer baru
    let mut model = BertForSequenceClassification::new(&vs.root(), &config)?;
    model.classifier = nn::linear(
        &vs.root() / "classifier",
        hidden_size,
        num_labels,
        Default::default(),
    );

    // Muat bobot model jika perlu
    vs.load("path/bert_model.ot")?;

    // Debug print config hasil parsing
    println!("{:?}", config);
    // Modify num_labels in config
    //let config_json = std::fs::read_to_string(&config_path)?;
    //let mut config_value: serde_json::Value = serde_json::from_str(&config_json)?;
    //config_value["num_labels"] =
    //serde_json::Value::Number(serde_json::Number::from(label_map.len() as i64));
    //let config: BertConfig = serde_json::from_value(config_value)?;

    // Initialize VarStore and model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let model = BertForSequenceClassification::new(&vs.root(), &config)?;
    vs.load(weights_path)?;

    let mut optimizer = nn::Adam::default().build(&vs, 3e-5)?;

    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;

    let messages = records
        .iter()
        .map(|r| r.message.clone())
        .collect::<Vec<_>>();
    let labels_encoded: Vec<i64> = records
        .iter()
        .map(|r| *label_map.get(&r.label).unwrap())
        .collect();

    let max_len = 64;
    let batch_size = 8;
    let epochs = 3;

    let batched_texts = batchify(&messages, batch_size);
    let batched_labels = batchify(&labels_encoded, batch_size);

    for epoch in 1..=epochs {
        let start = Instant::now();

        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_examples = 0;

        for (batch_texts, batch_labels) in batched_texts.iter().zip(batched_labels.iter()) {
            let encodings =
                tokenizer.encode_list(&batch_texts, max_len, &TruncationStrategy::LongestFirst, 0);

            let input_ids: Vec<Vec<i64>> = encodings.iter().map(|e| e.token_ids.clone()).collect();
            let attention_mask: Vec<Vec<i64>> = input_ids
                .iter()
                .map(|ids| ids.iter().map(|&id| if id != 0 { 1 } else { 0 }).collect())
                .collect();

            let input_ids = Tensor::stack(
                &input_ids
                    .iter()
                    .map(|ids| Tensor::from_slice(ids).to_device(device))
                    .collect::<Vec<_>>(),
                0,
            );
            let attention_mask = Tensor::stack(
                &attention_mask
                    .iter()
                    .map(|mask| Tensor::from_slice(mask).to_device(device))
                    .collect::<Vec<_>>(),
                0,
            );
            let label_tensor = Tensor::from_slice(&batch_labels).to_device(device);

            optimizer.zero_grad();
            let output = model.forward_t(
                Some(&input_ids),
                Some(&attention_mask),
                None,
                None,
                None,
                true,
            );

            let logits = output.logits;
            let loss = logits.cross_entropy_for_logits(&label_tensor);
            loss.backward();
            optimizer.step();

            total_loss += loss.double_value(&[]);
            total_examples += batch_labels.len();

            let predicted = logits.argmax(-1, false);
            let correct = predicted
                .eq_tensor(&label_tensor)
                .to_kind(tch::Kind::Int64)
                .sum(tch::Kind::Int64);
            total_correct += correct.int64_value(&[]);
        }
        let duration = start.elapsed();

        println!(
            "Epoch {} selesai dalam {:.2} detik. Avg Loss: {:.4}, Accuracy: {:.2}%",
            epoch,
            duration.as_secs_f64(),
            total_loss / total_examples as f64,
            (total_correct as f64 / total_examples as f64) * 100.0
        );
    }

    println!("Training selesai.");

    Ok(())
}
