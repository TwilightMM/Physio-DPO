# Physio-DPO

Physio-DPO is a protein sequence optimization pipeline based on **DPO (Direct Preference Optimization)** with additional **physio-inspired energy gap weighting**.

This repository contains:
- Dataset construction (generate sequences, fold structures, score, and build preference pairs)
- Conversion from the preference dataset to a DPO-ready JSON format
- DPO training and inference scripts

## Repository Layout

- `PhysioDPO_Data/PhysioDPO_Data/`
  - `main_physio_dpo.py`: end-to-end dataset construction pipeline
  - `modules/`: generators / folders (folding) / scorers / pairers
  - `readme.md`: output folder structure description (generated artifacts)
- `PhysioDPO_Train/PhysioDPO_Train/`
  - `scripts/convert_data_for_training.py`: convert `preference_dataset.jsonl` -> DPO JSON
  - `scripts/download_model.py`: download HuggingFace model files (optional)
  - `scripts/train_single_gpu.sh`, `scripts/train_multi_gpu.sh`: training entrypoints
  - `scripts/inference.sh`: run inference for generated sequences + evaluation
- `Protein_Case/`
  - `display.py`: generate a simple local HTML visualization for a PDB case

## Notes About Large Files

Model weights, checkpoints, HuggingFace caches, and generated outputs are excluded from Git tracking via `.gitignore`.
You should generate/download them locally.

## Prerequisites

### 1) Python dependencies

The main training dependencies are in:
- `PhysioDPO_Train/PhysioDPO_Train/requirements.txt`

Install example:

```bash
pip install -r PhysioDPO_Train/PhysioDPO_Train/requirements.txt
```

### 2) GPU

Training uses CUDA. For best results, ensure you have a working CUDA + PyTorch setup.

## Workflow

### Step A. Download the base model (optional)

You can pre-download base model files using:

```bash
python PhysioDPO_Train/PhysioDPO_Train/scripts/download_model.py --model_id hugohrban/progen2-xlarge --local_dir ./models/progen2-xlarge
```

If you are in a region where the HuggingFace mirror works better, add `--mirror`.

### Step B. Build the preference dataset

Run the dataset construction pipeline:

```bash
python PhysioDPO_Data/PhysioDPO_Data/main_physio_dpo.py --output_dir ./output --n_samples 10 --n_mutants 2
```

This generates (inside `--output_dir`):
- `scored_sequences.jsonl`
- `preference_dataset.jsonl`
- `pdbs/`, `pdbs_mutants/`, and optional `visualizations/`

### Step C. Convert preference dataset to DPO training format

```bash
python PhysioDPO_Train/PhysioDPO_Train/scripts/convert_data_for_training.py \
  --input ../PhysioDPO_Data/PhysioDPO_Data/output/preference_dataset.jsonl \
  --output PhysioDPO_Train/PhysioDPO_Train/data/demo.json
```

The output format matches what the DPO training code expects.

### Step D. Train with DPO

Single GPU:

```bash
bash PhysioDPO_Train/PhysioDPO_Train/scripts/train_single_gpu.sh
```

Multi-GPU (DeepSpeed):

```bash
bash PhysioDPO_Train/PhysioDPO_Train/scripts/train_multi_gpu.sh
```

### Step E. Inference

```bash
bash PhysioDPO_Train/PhysioDPO_Train/scripts/inference.sh
```

## Visualization

To generate a local HTML visualization for a PDB case:

```bash
python Protein_Case/display.py
```

The script will attempt to download `3Dmol-min.js` locally; if that fails, it falls back to a CDN URL.

## Questions / Issues

If you run into compatibility issues with TRL versions or model loading, please open an issue and include:
- your PyTorch / CUDA version
- the TRL version (`pip show trl`)
- the full error log

