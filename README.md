# Simple Self-Distillation
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.01193-b31b1b.svg)](https://arxiv.org/abs/2604.01193)
[![License](https://img.shields.io/badge/License-Apple-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

### Embarrassingly Simple Self-Distillation Improves Code Generation

Ruixiang Zhang\*, Richard He Bai\*, Huangjie Zheng\*, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang\*

<sub>\*Equal contribution</sub>

</div>

<p align="center">
  <img src="figures/fig_teaser.png" width="100%" alt="SSD Overview">
</p>

## ✨ Overview

This repository reproduces the method from the paper:

> **Embarrassingly Simple Self-Distillation Improves Code Generation**

The approach consists of three simple steps:

1. **Sample** solutions from a frozen model at non-unit temperature  
2. **Fine-tune** on raw, unverified outputs using standard cross-entropy  
3. **Decode** with a separately tuned temperature  

**No rewards · No verifier · No teacher · No RL**

For full details, see the [paper](https://arxiv.org/abs/2604.01193).

---

## 📰 News

- **[2026-04-03]** 🚀 Initial release of repository  
- **[2026-04-03]** 🤗 Model checkpoints coming soon on Hugging Face
- **[2026-04-07]** 🤗 Model checkpoints released
- *(More updates will be added here)*

---

## 🚀 Getting Started

```bash
git clone https://github.com/apple/ml-ssd.git
cd ml-ssd
uv sync --group evaluation
```


<details>
<summary>Evaluation commands</summary>

```bash
source .venv/bin/activate
python evaluation/eval.py \
    --model <hf_model_name> \
    --tensor_parallel_size 4 \
    --max_tokens 65536 \
    --n_repeat 10 \
    --sampling_params "temperature=0.9,top_p=0.8,top_k=20" \
    --output_path ./results/
```

> **Note:** The sampling parameters above are illustrative. Please refer to each model's HuggingFace model card for the recommended sampling parameters.

</details>

## 🤗 Models
| Model | HuggingFace |
|:---|:---|
| SSD-4B-Instruct | [apple/SimpleSD-4B-instruct](https://huggingface.co/apple/SimpleSD-4B-instruct) |
| SSD-4B-Thinking | [apple/SimpleSD-4B-thinking](https://huggingface.co/apple/SimpleSD-4B-thinking) |
| SSD-30B-A3B-Instruct | [apple/SimpleSD-30b-a3b-instruct](https://huggingface.co/apple/SimpleSD-30B-instruct) |

## 📁 Repository Structure

```
├── evaluation/
│   ├── eval.py                  # CLI entry point
│   ├── benchmark.py             # LiveCodeBench v6 implementation
│   └── livecodebench_utils.py   # Code execution utilities
├── figures/
│   └── fig_teaser.png
├── pyproject.toml
└── README.md
```

## 📝 Citation

```bibtex
@misc{zhang2026embarrassinglysimpleselfdistillationimproves,
      title={Embarrassingly Simple Self-Distillation Improves Code Generation},
      author={Ruixiang Zhang and Richard He Bai and Huangjie Zheng and Navdeep Jaitly and Ronan Collobert and Yizhe Zhang},
      year={2026},
      eprint={2604.01193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.01193},
}
```
