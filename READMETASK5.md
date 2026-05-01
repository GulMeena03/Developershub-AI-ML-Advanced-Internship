# Auto Tagging Support Tickets Using LLM

## Objective

The goal of this project is to automatically classify customer support tickets into predefined categories using Large Language Models (LLMs). Six categories are considered:

- `billing`
- `technical`
- `account`
- `product_inquiry`
- `complaint`
- `feature_request`

The task simulates a real‑world support system where incoming tickets must be routed correctly to the appropriate department or handled by automated workflows. We compare three increasingly sophisticated approaches: **zero‑shot classification**, **few‑shot learning** (5 examples per category), and **fine‑tuning** a pre‑trained transformer model.

---

## Methodology / Approach

### 1. Dataset Generation
Since real labeled ticket data is often proprietary, we synthetically generate 1,500 tickets using category‑specific templates. Each template includes placeholders (dates, invoice numbers, error codes, etc.) that are filled randomly. To mimic real‑world noise:
- 80% of tickets receive random character swaps (typo simulation).
- 5% of tickets are randomly relabeled to a different category (label noise).

The dataset is split into **Train (1,200)** and **Test (300)** with stratified sampling.

### 2. Models & Techniques

| Method | Description | Model Used |
|--------|-------------|-------------|
| **Zero‑shot** | Predict categories by natural language inference without any training examples. | `typeform/distilbert-base-uncased-mnli` (pipeline) |
| **Few‑shot** | Encode 5 example tickets per category into prototype vectors via Sentence‑BERT. New tickets are assigned to the most similar prototype (cosine similarity). | `sentence-transformers/all-MiniLM-L6-v2` |
| **Fine‑tuned** | Standard supervised fine‑tuning of a classifier on the training set (3 epochs, batch size 16, max length 256). | `distilbert-base-uncased` with classification head |

### 3. Evaluation Metrics
- Top‑1 accuracy
- Top‑3 accuracy (whether the true label appears in the top 3 predicted categories)
- Per‑class precision, recall, F1‑score

---

## Key Results / Observations

| Method | Top‑1 Accuracy | Top‑3 Accuracy |
|--------|----------------|----------------|
| Zero‑shot | 31.0% | 62.3% |
| Few‑shot (5‑shot) | 81.3% | 94.3% |
| Fine‑tuned (3 epochs) | **96.3%** | **99.0%** |

### Detailed Observations
- **Zero‑shot** performs poorly (31% top‑1). It struggles especially with `technical` and `feature_request` categories (0% recall). This is expected because NLI models are not tuned for fine‑grained intent classification with domain‑specific templates.
- **Few‑shot** achieves a strong 81.3% top‑1 accuracy with only 5 examples per class. The top‑3 accuracy of 94.3% shows that the correct category is almost always among the top three predictions – very useful for human‑in‑the‑loop triage.
- **Fine‑tuning** reaches near‑perfect performance (96.3% top‑1, 99% top‑3). The per‑class F1 scores are balanced (0.93–0.99), demonstrating that even a small DistilBERT model can master this task when provided with sufficient labeled data.

### Practical Takeaways
1. **Zero‑shot** is not reliable for production ticket tagging (unless categories are extremely distinct).
2. **Few‑shot learning** is a viable low‑resource alternative – easy to implement and requires no model training.
3. **Fine‑tuning** yields the best results, but needs a few hundred labeled examples per class. The effort of labeling tickets pays off significantly.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install transformers datasets sentence-transformers scikit-learn torch accelerate