# SIGMA-POI

SIGMA-POI is a Linux-first research pipeline for semantic-ID based next-POI recommendation:

- POI embedding
- cosine+EMA SID training/export
- SID-LLM alignment
- SFT backend orchestration
- GRPO backend orchestration with `ms-swift`
- retrieval-based history construction
- local batch inference and evaluation
- clean Linux-first CLI without legacy command compatibility

## Layout

- `src/gnprsid/`: core package; the import namespace is kept stable for existing scripts
- `configs/`: YAML configs
- `data/`: imported and prepared dataset assets
- `artifacts/`: SID, retrieval, and alignment artifacts
- `checkpoints/`: training outputs
- `outputs/`: predictions, metrics, and summaries
- `docs/`: project documents

## Dataset

This repository provides the Foursquare-NYC dataset for convenient demonstration and evaluation. The NYC raw data is placed under `data/NYC/raw/`, and the project CLI can prepare the processed samples, semantic-ID artifacts, retrieval assets, predictions, and evaluation outputs used by the examples below.

## Environment and Frameworks

SIGMA-POI is tested as a Linux-first training pipeline and requires Python 3.10 or newer. GPU acceleration is strongly recommended for model training, alignment, SFT, and GRPO runs.

Core Python dependencies are declared in `pyproject.toml`, including `torch`, `transformers`, `datasets`, `peft`, `trl`, `sentence-transformers`, `numpy`, `pandas`, `scikit-learn`, `pyarrow`, and related utilities. The main training frameworks are:

- `TRL` + `PEFT` for semantic-ID alignment SFT and LoRA adapter training
- `LLaMA-Factory` for recommendation-task SFT / warmup training through `llamafactory-cli`
- `ms-swift` for GRPO / RLHF training through the `swift` CLI
- `Transformers` and `PyTorch` for model loading, inference, adapter merging, and local training components
- optional `vLLM`, FlashAttention, `accelerate`, and `wandb` support for faster or logged multi-GPU training runs

## Primary CLI

```bash
python -m gnprsid.cli data import-legacy --dataset NYC --legacy-root /path/to/old/SIGMA-POI
python -m gnprsid.cli data prepare-nyc --dataset NYC
python -m gnprsid.cli sid train --config configs/train/sid_nyc.yaml
python -m gnprsid.cli sid export --config configs/train/sid_nyc.yaml
python -m gnprsid.cli alignment build-data --dataset NYC --semantic-schema semantic_spatial_v2 --grid-size 8 --split-by abc
python -m gnprsid.cli train run --stage alignment --config configs/train/alignment_phase_a.yaml
python -m gnprsid.cli train merge-peft --model-config configs/models/qwen3_8b.yaml --adapter-path checkpoints/NYC/alignment/qwen25_7b_phase_a/final
python -m gnprsid.cli train run --stage alignment --config configs/train/alignment_phase_b1.yaml
python -m gnprsid.cli train merge-peft --model-config configs/models/qwen3_8b.yaml --adapter-path checkpoints/NYC/alignment/qwen25_7b_phase_b1/final
python -m gnprsid.cli train run --stage alignment --config configs/train/alignment_phase_b2.yaml
python -m gnprsid.cli train merge-peft --model-config configs/models/qwen3_8b.yaml --adapter-path checkpoints/NYC/alignment/qwen25_7b_phase_b2/final
python -m gnprsid.cli alignment evaluate --dataset NYC --model-config configs/models/qwen25_7b.yaml --task sid_to_abc_profile
python -m gnprsid.cli grpo build-data --dataset NYC --model-profile qwen3-8b-instruct
python -m gnprsid.cli train run --stage grpo --config configs/train/grpo_ms_swift_qwen3.yaml
python -m gnprsid.cli train merge-peft --model-config configs/models/qwen3_8b.yaml --adapter-path checkpoints/NYC/grpo/qwen3_8b_sid_current/checkpoint-100
python -m gnprsid.cli retrieval build-bank --dataset NYC --repr sid
python -m gnprsid.cli retrieval build-similar --dataset NYC --repr sid --split test --config configs/retrieval/default.yaml
python -m gnprsid.cli infer batch --dataset NYC --repr sid --history-source current --model-config configs/models/qwen3_8b.yaml --decoding-mode direct
python -m gnprsid.cli eval run --predictions outputs/NYC/predictions/run.json
python -m gnprsid.cli eval summarize --dataset NYC
```


## Acknowledgements

Part of this codebase is designed based on [wds1996/GNPR-SID](https://github.com/wds1996/GNPR-SID). We thank the original authors for releasing their implementation and research artifacts.
