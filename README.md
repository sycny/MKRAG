# MKRAG


This repository contains code for our AMIA'2024 paper "MKRAG: Medical Knowledge Retrieval Augmented Generation for Medical Question Answering".
![MKRAG](https://github.com/sycny/sycny.github.io/blob/main/images/AMIA.png)
## Overview

The system combines:
- Large Language Models (Vicuna-7B) for answer generation
- Knowledge Graph integration for medical facts
- Contriever for relevant fact retrieval

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
medicalQA/
├── Release/
│   ├── main.py          # Main execution script
│   ├── generator.py     # Model generation wrapper
│   └── utils.py         # Utility functions
├── data/
│   ├── ddb/            # DrugBank database files
│   │   ├── ddb_names.json
│   │   └── ddb_relas.json
│   └── [Other data directories...]
└── [Other project directories...]
```

## Data Preparation

1. Download required model weights:
   - Vicuna-7B model: We used Vicuna-7B-v1 in our research. Due to licensing restrictions, we cannot publicly distribute the model. If you would like to reproduce our results, please contact [us](yucheng.shi (AT) uga (DOT) edu) via email for guidance and access instructions.
   - SapBERT/Contriever model weights are automatically downloaded from Huggingface.

2. Prepare knowledge graph data:
   - Place DrugBank database files in `data/ddb/`
   - You can find more information [here](https://arxiv.org/abs/2104.06378).

## Running the Code

### Basic Usage

```bash
python main.py --model vicuna --device cuda 
```

### Key Arguments

- `--model`: Model type ('vicuna', 'gpt2', etc.)
- `--device`: Computing device ('cuda' or 'cpu')
- `--path`: Path to medical QA dataset
- `--fact_number`: Number of facts to retrieve (default: 8)
- `--loademb`: Whether to load pre-computed embeddings. You can download it [here](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EdxSD4AlEtlKr1OQp4WUwowBpLj1KV76QpLw7XPfpJu7WQ?e=Uhuqpj).

## Model Architecture

### 1. Knowledge Retrieval
- Uses Contriever for retrieving relevant medical facts
- Integrates DrugBank knowledge graph

### 2. Answer Generation
- Employs Vicuna-7B for generating answers
- Incorporates retrieved knowledge into prompts

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shi2023mededit,
  title={Mededit: Model editing for medical question answering with external knowledge bases},
  author={Shi, Yucheng and Xu, Shaochen and Liu, Zhengliang and Liu, Tianming and Li, Xiang and Liu, Ninghao},
  journal={arXiv preprint arXiv:2309.16035},
  year={2023}
}
```
