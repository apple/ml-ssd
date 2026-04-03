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

This repo is for paper reproduction for improving code generation in three steps: (1) **Sample** solutions from the frozen model at non-unit temperature, (2) **Fine-tune** on those raw, unverified outputs via standard cross-entropy, and (3) **Decode** at a separately tuned temperature. No rewards, no verifier, no teacher, no RL. See the [paper](https://arxiv.org/abs/2604.01193) for full details.



## News
* 04/03/2026 Model checkpoints will be released on HF shortly. 


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
> Note: Model checkpoints are coming soon. Stay tuned!

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
