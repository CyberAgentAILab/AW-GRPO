# Auto-Weighted Group Relative Preference Optimization for Multi-Objective Text Generation Task

Implementation of **Auto-Weighted Group Relative Preference Optimization (AW-GRPO)** for English→Japanese MT.

# Environment
- Python 3.12, CUDA 12.6
- Base image example: linux/amd64 `nvidia/cuda:12.6.1-devel-ubuntu22.04`

# Quick Start

### 1. Setup
```bash
python3 -m venv env && source env/bin/activate

# system deps + python deps
bash setup.sh
```

### 2. Get data (WMT En→Ja)
```bash
# files will be placed under dataset/wmt.en-ja/
bash get_wmt.sh
```

### 3. Train
```bash
bash run_aw_grpo.sh \
  --output_dir aw_grpo_results \
  --reward_functions combined
```


# Cite
```bash
@inproceedings{ichihara-etal-2025-awgrpo,
    title = "Auto-Weighted Group Relative Preference Optimization for Multi-Objective Text Generation Tasks",
    author = "Yuki, Ichihara and Yuu, Jinnai",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    year = "2025",
}
```
