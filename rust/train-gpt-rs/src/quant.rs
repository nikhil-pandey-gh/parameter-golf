use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub shape: Vec<usize>,
    pub scheme: QuantScheme,
    pub data: Vec<i8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum QuantScheme {
    PerTensor { scale: f32 },
    PerRow { scales: Vec<f32> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedState {
    pub format: String,
    pub tensors: BTreeMap<String, QuantizedTensor>,
}

#[derive(Debug, Clone)]
pub struct QuantStats {
    pub tensor_count: usize,
    pub baseline_bytes: usize,
    pub int8_bytes: usize,
}

pub fn quantize_varmap(varmap: &VarMap) -> Result<(QuantizedState, QuantStats)> {
    let lock = varmap.data().lock().unwrap();
    let mut tensors = BTreeMap::new();
    let mut stats = QuantStats {
        tensor_count: 0,
        baseline_bytes: 0,
        int8_bytes: 0,
    };
    for (name, var) in lock.iter() {
        let tensor = var.as_tensor().flatten_all()?.to_dtype(DType::F32)?;
        let values = tensor.to_vec1::<f32>()?;
        let shape = var.shape().dims().to_vec();
        stats.tensor_count += 1;
        stats.baseline_bytes += values.len() * std::mem::size_of::<f32>();
        let qt = if shape.len() == 2 {
            quantize_per_row(shape, values)?
        } else {
            quantize_per_tensor(shape, values)
        };
        stats.int8_bytes += qt.data.len();
        stats.int8_bytes += match &qt.scheme {
            QuantScheme::PerTensor { .. } => std::mem::size_of::<f32>(),
            QuantScheme::PerRow { scales } => scales.len() * std::mem::size_of::<f32>(),
        };
        tensors.insert(name.clone(), qt);
    }
    Ok((
        QuantizedState {
            format: "int8_clean_per_row_v1".into(),
            tensors,
        },
        stats,
    ))
}

pub fn save_quantized_state(state: &QuantizedState, path: &Path) -> Result<()> {
    let json = serde_json::to_vec(state)?;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    use std::io::Write as _;
    encoder.write_all(&json)?;
    let bytes = encoder.finish()?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn load_quantized_state(path: &Path) -> Result<QuantizedState> {
    let bytes = fs::read(path)?;
    let mut decoder = ZlibDecoder::new(bytes.as_slice());
    let mut json = Vec::new();
    use std::io::Read as _;
    decoder.read_to_end(&mut json)?;
    Ok(serde_json::from_slice(&json)?)
}

pub fn restore_quantized_state(
    varmap: &VarMap,
    state: &QuantizedState,
    device: &Device,
) -> Result<()> {
    let mut lock = varmap.data().lock().unwrap();
    for (name, qt) in &state.tensors {
        let Some(var) = lock.get_mut(name) else {
            continue;
        };
        let values = match &qt.scheme {
            QuantScheme::PerTensor { scale } => qt
                .data
                .iter()
                .map(|v| f32::from(*v) * *scale)
                .collect::<Vec<_>>(),
            QuantScheme::PerRow { scales } => {
                ensure!(
                    qt.shape.len() == 2,
                    "per-row quantization requires rank-2 tensors"
                );
                let rows = qt.shape[0];
                let cols = qt.shape[1];
                ensure!(rows == scales.len(), "row scale count mismatch for {name}");
                let mut out = vec![0_f32; qt.data.len()];
                for row in 0..rows {
                    let scale = scales[row];
                    for col in 0..cols {
                        let index = row * cols + col;
                        out[index] = f32::from(qt.data[index]) * scale;
                    }
                }
                out
            }
        };
        let tensor = Tensor::from_vec(values, qt.shape.clone(), device)?;
        var.set(&tensor)?;
    }
    Ok(())
}

fn quantize_per_tensor(shape: Vec<usize>, values: Vec<f32>) -> QuantizedTensor {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let data = values
        .into_iter()
        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    QuantizedTensor {
        shape,
        scheme: QuantScheme::PerTensor { scale },
        data,
    }
}

fn quantize_per_row(shape: Vec<usize>, values: Vec<f32>) -> Result<QuantizedTensor> {
    let rows = shape[0];
    let cols = shape[1];
    ensure!(
        rows * cols == values.len(),
        "rank-2 tensor shape does not match flattened payload"
    );
    let mut scales = Vec::with_capacity(rows);
    let mut data = Vec::with_capacity(values.len());
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_values = &values[start..end];
        let max_abs = row_values
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales.push(scale);
        data.extend(
            row_values
                .iter()
                .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8),
        );
    }
    Ok(QuantizedTensor {
        shape,
        scheme: QuantScheme::PerRow { scales },
        data,
    })
}
