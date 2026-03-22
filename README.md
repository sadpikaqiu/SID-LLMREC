# GNPR-SID

Linux-first refactor of GNPR-SID built around the V2 pipeline:

- POI embedding
- cosine+EMA SID training/export
- SID-LLM alignment
- SFT backend orchestration
- retrieval-based history construction
- local batch inference and evaluation
- clean Linux-first CLI without legacy command compatibility

## Layout

- `src/gnprsid/`: core package
- `configs/`: YAML configs
- `data/`: imported and prepared dataset assets
- `artifacts/`: SID, retrieval, and alignment artifacts
- `checkpoints/`: training outputs
- `outputs/`: predictions, metrics, and summaries

## Primary CLI

```bash
python -m gnprsid.cli data import-legacy --dataset NYC --legacy-root /path/to/old/GNPR-SID
python -m gnprsid.cli data prepare-nyc --dataset NYC
python -m gnprsid.cli sid train --config configs/train/sid_nyc.yaml
python -m gnprsid.cli sid export --config configs/train/sid_nyc.yaml
python -m gnprsid.cli alignment build-data --dataset NYC
python -m gnprsid.cli train run --stage alignment --config configs/train/alignment_trl.yaml
python -m gnprsid.cli retrieval build-bank --dataset NYC --repr sid
python -m gnprsid.cli retrieval build-similar --dataset NYC --repr sid --split test --config configs/retrieval/default.yaml
python -m gnprsid.cli infer batch --dataset NYC --repr sid --history-source retrieval --model-config configs/models/qwen25_7b.yaml
python -m gnprsid.cli eval run --predictions outputs/NYC/predictions/run.json
python -m gnprsid.cli eval summarize --dataset NYC
```

## Recommended Linux Order

```bash
export PYTHONPATH=$PWD/src

python -m gnprsid.cli data import-legacy --dataset NYC --legacy-root /path/to/legacy/GNPR-SID
python -m gnprsid.cli data prepare-nyc --dataset NYC --current-k 49

python -m gnprsid.cli sid train --config configs/train/sid_nyc.yaml
python -m gnprsid.cli sid export --config configs/train/sid_nyc.yaml

python -m gnprsid.cli data prepare-nyc --dataset NYC --current-k 49 \
  --sid-map-path artifacts/NYC/sid/pid_to_sid.json

python -m gnprsid.cli alignment build-data --dataset NYC \
  --sid-map-path artifacts/NYC/sid/pid_to_sid.json

python -m gnprsid.cli train run --stage alignment --config configs/train/alignment_trl.yaml
python -m gnprsid.cli train run --stage sft --config configs/train/sft_llamafactory.yaml

python -m gnprsid.cli retrieval build-bank --dataset NYC --repr sid
python -m gnprsid.cli retrieval build-similar --dataset NYC --repr sid --split test \
  --config configs/retrieval/default.yaml --model-config configs/models/qwen25_7b.yaml

python -m gnprsid.cli infer batch --dataset NYC --repr sid --history-source retrieval \
  --model-config configs/models/qwen25_7b.yaml \
  --checkpoint-path checkpoints/NYC/sft/qwen25_7b_sid_current/llamafactory_output

python -m gnprsid.cli eval run \
  --predictions outputs/NYC/predictions/sid_retrieval_test.json

python -m gnprsid.cli eval summarize --dataset NYC
```

## Notes

- The legacy project stays untouched. Import once, then operate entirely inside this project.
- Official retrieval defaults use `Qwen2.5-7B-Instruct + bf16 + mean pooling + max_length=2048`.
- Alignment data follows the V2 bidirectional design: `POI attributes -> SID` and `SID -> POI attributes`.
- NYC import enriches `poi_info.csv` with latitude/longitude from the raw `NYC.txt` check-in table, so alignment data includes category, region, latitude, longitude, visit hours, visit count, and unique user count.
- `GRPO` is reserved as a backend interface but not implemented in this first version.
