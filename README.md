<div align="center">
<h1>SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning</h1>
</div>

<div align="center" style="display: flex; gap: 5px; justify-content: center;">
<a href="https://idea-finai.github.io/SQL-R1/"><img src="https://img.shields.io/badge/🏠_Homepage-SQL--R1-4B4B77?style=flat-square"/></a>
<a href="https://arxiv.org/abs/2504.08600"><img src="https://img.shields.io/badge/📑_arXiv-2504.08600-00A98F?style=flat-square"/></a>
<a href="https://github.com/MPX0222/SQL-R1"><img src="https://img.shields.io/badge/⭐_GitHub-SQL--R1-2F80ED?style=flat-square"/></a>
<a href="https://huggingface.co/MPX0222forHF/SQL-R1-7B"><img src="https://img.shields.io/badge/🤗_HuggingFace-SQL--R1-FF9D00?style=flat-square"/></a>
<a href="https://github.com/MPX0222/SQL-R1/stargazers"><img src="https://img.shields.io/github/stars/MPX0222/SQL-R1?style=flat-square&color=946CE6"/></a>
</div>


> 🚧 **Note:** This repository is under active development. We will be continuously updating with model weights, training code, and more resources in the coming weeks. Stay tuned! ✨

## 📖 Overview

Natural Language to SQL (NL2SQL) enables intuitive interactions with databases by transforming natural language queries into structured SQL statements.  Despite recent advancements in enhancing human-computer interaction within database applications, significant challenges persist, particularly regarding the inference performance in complex scenarios involving multi-table joins and nested queries. Current methodologies primarily utilize supervised fine-tuning (SFT) to train the NL2SQL model, which may limit adaptability and interpretability in new environments (e.g., finance and healthcare). In order to enhance the reasoning performance of the NL2SQL model in the above complex situations, we introduce SQL-R1, a novel NL2SQL reasoning model trained by the reinforcement learning (RL) algorithms. We design a specialized RL-based reward function tailored for NL2SQL tasks and discussed the impact of cold start on the effectiveness of intensive training. In addition, we achieve competitive accuracy using only a tiny amount of synthetic NL2SQL data for augmented training and further explore data engineering for RL. In existing experiments, SQL-R1 achieves execution accuracy of 88.6\% and 67.1\% on the benchmark Spider and BIRD, respectively.

<div align="center">
<img src="images/overview.png" alt="SQL-R1 Overview" width="800"/>
</div>


## 📚 Citations

```bibtex
@article{ma2025sql,
  title={SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning},
  author={Ma, Peixian and Zhuang, Xialie and Xu, Chengjin and Jiang, Xuhui and Chen, Ran and Guo, Jian},
  journal={arXiv preprint arXiv:2504.08600},
  year={2025}
}
```

## 📰 News

- **[2024.05.27]** 🎉 We have released the full code of SQL-R1.
- **[2024.05.21]** 🎉 We have released our model weights on Hugging Face! Check out the [Model Weights](#-model-weights) section below.
- **[2024.04.11]** 📑 Our paper is now available on [arXiv](https://arxiv.org/abs/2504.08600).


## 🚀 Coming Soon Checklist

- [x] 📊 Release model weights on HuggingFace
- [x] 🔧 Open source training code
- [x] 📝 Detailed documentation
- [x] 🛠️ Environment setup guide


## 🤖 Model Weights

We are excited to release our SQL-R1 model weights! You can find them on Hugging Face:

| Model  | Size | Link |
|-------------|-------------|------|
| SQL-R1 (3B) | 3B | [🤗 Download](https://huggingface.co/MPX0222forHF/SQL-R1-3B) |
| SQL-R1 (7B) | 7B | [🤗 Download](https://huggingface.co/MPX0222forHF/SQL-R1-7B) |
| SQL-R1 (14B) | 14B | [🤗 Download](https://huggingface.co/MPX0222forHF/SQL-R1-14B) |


## 📑 Documentation Structure

This repository is organized as follows:

```
SQL-R1/
├── data/                             # Data processing scripts and datasets
│   ├── Spider/      
│   └── BIRD/        
├── models/                           # Base models
│   ├── Qwen2.5-Coder-3B-Instruct/   
│   └── Qwen2.5-Coder-7B-Instruct/   
├── sh/                               # Scripts for training, inference and evaluation
├── src/                              # Source code
└── verl/                             # Verl framework
```


## 🛠️ Environment Setup

### Prerequisites
- Python 3.9+
- CUDA 12.0+
- 8 x 80GB+ GPU (for training) / 2 x 40GB GPU (for inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MPX0222/SQL-R1.git
cd SQL-R1
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n sqlr1 python=3.9
```

3. Install dependencies:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 ray
pip install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib sqlparse func_timeout
```

4. Download the model weights from Hugging Face and put them in the `models/` directory

5. Copy the database information in the `db_info` directory to the related dataset (`data/Spider/`, `data/BIRD/`) directory.

## 🚀 Quick Start
> Note: Please set the related data paths and params before running the scripts.

1. Run training:
```bash
sh sh/train_sqlr1.sh
```

2. Run inference:
```bash
sh sh/inference.sh"
```

3. Run evaluation:
```bash
# evaluate spider
sh sh/eval_spider.sh
# evaluate bird
sh sh/eval_bird.sh
```

## Thanks for

We thank [OmniSQL](https://github.com/GIST-IRR/OmniSQL) and follow their evaluation code and database information retrieval code. We have adapted and modified their evaluation scripts for our project.
