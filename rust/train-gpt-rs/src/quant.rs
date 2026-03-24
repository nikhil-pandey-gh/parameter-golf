use std::cmp::Ordering;
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

const INT8_KEEP_FLOAT_FP32_NAME_PATTERNS: &[&str] = &[
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "skip_weights",
];
const INT8_KEEP_FLOAT_MAX_NUMEL: usize = 65_536;
const INT8_CLIP_Q: f32 = 0.999_998_4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum QuantScale {
    Scalar { value: f32 },
    PerRow { values: Vec<f32> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTensor {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantMeta {
    pub scheme: String,
    pub axis: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedState {
    pub format: String,
    pub quantized: BTreeMap<String, QuantizedTensor>,
    pub scales: BTreeMap<String, QuantScale>,
    pub dtypes: BTreeMap<String, String>,
    pub passthrough: BTreeMap<String, StoredTensor>,
    pub passthrough_orig_dtypes: BTreeMap<String, String>,
    pub qmeta: BTreeMap<String, QuantMeta>,
}

#[derive(Debug, Clone)]
pub struct QuantStats {
    pub tensor_count: usize,
    pub baseline_bytes: usize,
    pub int8_bytes: usize,
}

pub fn quantize_varmap(varmap: &VarMap) -> Result<(QuantizedState, QuantStats)> {
    let lock = varmap.data().lock().unwrap();
    let mut quantized = BTreeMap::new();
    let mut scales = BTreeMap::new();
    let mut dtypes = BTreeMap::new();
    let mut passthrough = BTreeMap::new();
    let mut passthrough_orig_dtypes = BTreeMap::new();
    let mut qmeta = BTreeMap::new();
    let mut stats = QuantStats {
        tensor_count: 0,
        baseline_bytes: 0,
        int8_bytes: 0,
    };
    for (name, var) in lock.iter() {
        let shape = var.shape().dims().to_vec();
        let tensor = var.as_tensor().flatten_all()?.to_dtype(DType::F32)?;
        let values = tensor.to_vec1::<f32>()?;
        stats.tensor_count += 1;
        stats.baseline_bytes += values.len() * std::mem::size_of::<f32>();

        if values.len() <= INT8_KEEP_FLOAT_MAX_NUMEL {
            let stored = keep_float_tensor(name, &shape, &values)?;
            stats.int8_bytes += stored_bytes(&stored);
            if stored.dtype != "float32" {
                passthrough_orig_dtypes.insert(name.clone(), "float32".into());
            }
            passthrough.insert(name.clone(), stored);
            continue;
        }

        let (qt, scale, meta) = if shape.len() == 2 {
            quantize_per_row(shape.clone(), values)?
        } else {
            quantize_per_tensor(shape.clone(), values)?
        };
        stats.int8_bytes += qt.data.len();
        stats.int8_bytes += match &scale {
            QuantScale::Scalar { .. } => std::mem::size_of::<f32>(),
            QuantScale::PerRow { values } => values.len() * std::mem::size_of::<f32>(),
        };
        quantized.insert(name.clone(), qt);
        scales.insert(name.clone(), scale);
        dtypes.insert(name.clone(), "float32".into());
        if let Some(meta) = meta {
            qmeta.insert(name.clone(), meta);
        }
    }
    Ok((
        QuantizedState {
            format: "int8_clean_per_row_v1".into(),
            quantized,
            scales,
            dtypes,
            passthrough,
            passthrough_orig_dtypes,
            qmeta,
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
    for (name, qt) in &state.quantized {
        let Some(var) = lock.get_mut(name) else {
            continue;
        };
        let scale = state
            .scales
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing quant scale for {name}"))?;
        let values = match scale {
            QuantScale::Scalar { value } => qt
                .data
                .iter()
                .map(|v| f32::from(*v) * *value)
                .collect::<Vec<_>>(),
            QuantScale::PerRow { values: scales } => {
                ensure!(qt.shape.len() == 2, "per-row quantization requires rank-2 tensors");
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
    for (name, stored) in &state.passthrough {
        let Some(var) = lock.get_mut(name) else {
            continue;
        };
        let tensor = Tensor::from_vec(stored.data.clone(), stored.shape.clone(), device)?;
        var.set(&tensor)?;
    }
    Ok(())
}

fn keep_float_tensor(name: &str, shape: &[usize], values: &[f32]) -> Result<StoredTensor> {
    let store_fp32 = INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
        .iter()
        .any(|pattern| name.contains(pattern));
    let (dtype, data) = if store_fp32 {
        ("float32".to_string(), values.to_vec())
    } else {
        (
            "float16".to_string(),
            roundtrip_values(values, shape.to_vec(), DType::F16)?,
        )
    };
    Ok(StoredTensor {
        shape: shape.to_vec(),
        dtype,
        data,
    })
}

fn stored_bytes(stored: &StoredTensor) -> usize {
    let elem_size = match stored.dtype.as_str() {
        "float16" => 2,
        _ => 4,
    };
    stored.data.len() * elem_size
}

fn roundtrip_values(values: &[f32], shape: Vec<usize>, dtype: DType) -> Result<Vec<f32>> {
    let tensor = Tensor::from_vec(values.to_vec(), shape, &Device::Cpu)?;
    Ok(tensor.to_dtype(dtype)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn quantize_per_tensor(
    shape: Vec<usize>,
    values: Vec<f32>,
) -> Result<(QuantizedTensor, QuantScale, Option<QuantMeta>)> {
    let clip_abs = percentile_abs(&values);
    let scale = if clip_abs > 0.0 { clip_abs / 127.0 } else { 1.0 };
    let clipped = values
        .into_iter()
        .map(|value| (value.clamp(-clip_abs, clip_abs) / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Ok((
        QuantizedTensor {
            shape,
            data: clipped,
        },
        QuantScale::Scalar { value: scale },
        None,
    ))
}

fn quantize_per_row(
    shape: Vec<usize>,
    values: Vec<f32>,
) -> Result<(QuantizedTensor, QuantScale, Option<QuantMeta>)> {
    let rows = shape[0];
    let cols = shape[1];
    ensure!(rows * cols == values.len(), "rank-2 tensor shape does not match flattened payload");
    let mut scales = Vec::with_capacity(rows);
    let mut data = Vec::with_capacity(values.len());
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_values = &values[start..end];
        let clip_abs = percentile_abs(row_values);
        let scale = (clip_abs / 127.0).max(1.0 / 127.0);
        scales.push(scale);
        data.extend(row_values.iter().map(|value| {
            (value.clamp(-clip_abs, clip_abs) / scale)
                .round()
                .clamp(-127.0, 127.0) as i8
        }));
    }
    Ok((
        QuantizedTensor { shape, data },
        QuantScale::PerRow { values: scales },
        Some(QuantMeta {
            scheme: "per_row".into(),
            axis: 0,
        }),
    ))
}

fn percentile_abs(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut abs_values = values.iter().map(|value| value.abs()).collect::<Vec<_>>();
    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let last = abs_values.len() - 1;
    let position = INT8_CLIP_Q * last as f32;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        abs_values[lo]
    } else {
        let frac = position - lo as f32;
        abs_values[lo] * (1.0 - frac) + abs_values[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    #[test]
    fn percentile_matches_expected_interpolation() {
        let value = percentile_abs(&[0.0, 1.0, 2.0, 100.0]);
        assert!(value >= 2.0);
        assert!(value < 100.0);
    }

    #[test]
    fn small_control_tensors_stay_float() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _ = vb.get_with_hints((2, 3), "attn_scale", candle_nn::Init::Const(1.0))?;
        let (state, _) = quantize_varmap(&varmap)?;
        assert!(state.passthrough.contains_key("attn_scale"));
        assert!(!state.quantized.contains_key("attn_scale"));
        Ok(())
    }
}
