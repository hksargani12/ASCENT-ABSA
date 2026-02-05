# ASCENT: Interpretable Aspect-Based Sentiment Analysis via Contrastive Learning and Neuron-Level Optimization

### Overview
**ASCENT** is an interpretable framework for **Aspect-Based Sentiment Analysis (ABSA)** that combines **contrastive pretraining** with **neuron-level optimization**.  
The model enhances sentiment-aware representations through multi-objective learning and selectively fine-tunes neurons based on their attribution importance, improving both **accuracy** and **interpretability**.

This repository contains the code used for model pretraining, neuron localization, and selective fine-tuning experiments described in the study.

---

## ⚙️ Project Structure

```
ASCENT/
├── ascent_pretrain.py              # Multi-task pretraining with MLM and contrastive loss
├── ascent_neuron_localization.py   # Integrated Gradients-based neuron localization
├── ascent_neuron_analysis.py       # Segregates salient, dormant, and random neurons
├── ascent_finetuning.py            # Fine-tuning on selected neurons (salient, dormant, random)
│
├── custom_bert.py                  # Custom BERT with MLM + Sentiment classification heads
├── utils_features.py               # Common feature utility functions
├── utils.py                        # Loss functions and gradient masking utilities
├── evaluation.py                   # Model evaluation metrics
│
├── run_ascent_pretrain.sh          # Example pretraining command
├── run_ascent_finetuning.sh        # Example fine-tuning command
├── run_ascent_neuron_analysis.sh   # Example neuron analysis command
│
└── README.md
```

---

## 🧠 Datasets

ASCENT uses public ABSA datasets for the finetuning and two domain-level corpora for pretraining.

- **Pretraining datasets:** Yelp, Amazon  
- **Fine-tuning datasets:** SemEval Laptop 2014 (Lap14), SemEval Restaurant 2014 (Res14), and MAMS  

📂 **Dataset Access:**  
All datasets can be downloaded from the following Google Drive link:  
👉 **[Dataset Download (Google Drive)](https://drive.google.com/drive/folders/1xubtBXgXy_wzupXtVQjeNN1STMWtDpqS?usp=sharing)**  

---

## 🔧 Requirements

Ensure you have Python ≥3.8 and install all dependencies:

```bash
pip install -r requirements.txt
```

**Main packages:**
- `torch`
- `transformers`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jsonlines`
- `openpyxl`

---

## 🚀 Reproducing the Experiments

### 1️⃣ Pretraining ASCENT
Run the following command to train the ASCENT model using multi-task objectives:
```bash
bash run_ascent_pretrain.sh
```

This will generate a pre-trained checkpoint (e.g., `ascent_pretrain_yelp.bin`) combining MLM, sentiment classification, and contrastive objectives.

---

### 2️⃣ Neuron Localization
To compute neuron-level attributions via Integrated Gradients:
```bash
python ascent_neuron_localization.py   --data_path ./data/mams_test_dataset.json   --bert_model bert-base-uncased   --output_dir ./outputs/localization/   --output_prefix neurons_mams
```
This produces a `.jsonl` file containing per-layer neuron attribution scores.

---

### 3️⃣ Neuron Analysis
To extract salient, dormant, and random neurons:
```bash
bash run_ascent_neuron_analysis.sh
```
This step creates:
- `neurons_mams_salient.json`  
- `neurons_mams_dormant.json`  
- `neurons_mams_random.json`  
and generates an Excel file summarizing neuron distributions.

---

### 4️⃣ Fine-Tuning Modes
Fine-tune the ASCENT model using different neuron configurations:
```bash
bash run_ascent_finetuning.sh
```

**Available modes:**
- `--mode full` → Standard full-model fine-tuning  
- `--mode salient` → Fine-tune only salient neurons  
- `--mode dormant` → Fine-tune dormant neurons  
- `--mode random` → Fine-tune random neurons (control)

---

### 5️⃣ Evaluation
After each run, metrics are automatically saved to:
```
outputs/finetune/metrics.json
```

Metrics include:
- Sentiment accuracy, macro-precision, recall, F1
- MLM accuracy and perplexity

---

## 📊 Outputs Structure

```
outputs/
├── pretrain/
│   └── ascent_pretrain_yelp.bin
├── localization/
│   └── neurons_mams.sentiment.jsonl
├── analysis/
│   ├── neurons_mams_salient.json
│   ├── neurons_mams_dormant.json
│   ├── neurons_mams_random.json
│   └── neurons_mams_statistics.xlsx
└── finetune/
    ├── ascent_mams_full.bin
    ├── ascent_mams_salient.bin
    ├── ascent_mams_dormant.bin
    ├── ascent_mams_random.bin
    └── metrics.json
```

---

## 🧩 Notes

- Paths inside the bash scripts are placeholders; modify them to match your setup.  
- GPU execution is required (tested on CUDA 11.8 / PyTorch 2.x).  
- Neuron statistics and visualizations are auto-generated during analysis.  
- Dataset files are **not** included in this repository — download via the provided Drive link.

---

## 📜 Journal Submission Notice

This repository accompanies the manuscript submission of **ASCENT**.  
A detailed release, including:
- Full abstract and theoretical framework  
- Extended experimental results  
- Citation information (BibTeX)  
will be made public following publication.

---

<!-- ## ✉️ Contact

For academic correspondence or technical queries:  
**Muhammad Hasnain Khan**  
Iran University of Science and Technology, Tehran (Iran)  
Email: hksargani12@gmail.com 

--- -->
