# Cross-Lingual NER Adaptation

This repository presents a Cross-Lingual Named Entity Recognition (NER) Adaptation on Turkish using the `xlm-roberta-base` model.

---

## Phases

- Phase 1: EN and TR Zero-Shot results of XLM-R
- Phase 2: TR Zero-Shot results of XLM-R (fine-tuned on EN)
- Phase 3: TR adaptation results of XLM-R (fine-tuned on EN+TR)

---

## Evaluation Metrics of Phases

|   Phases      | EN-FT Samples | TR-FT Samples | English F1-Score | Turkish F1-Score |
|:-------------:|:-------------:|:-------------:|:----------------:|:----------------:|
|   Phase 1     |      0        |      0        |       0.72       |       0.63       |
|   Phase 2     |      20000    |      0        |       0.97       |       0.75       |
|   Phase 3.1   |      20000    |      250      |       0.97       |       0.81       |
|   Phase 3.2   |      20000    |      500      |       0.97       |       0.84       |
|   Phase 3.3   |      20000    |      1000     |       0.96       |       0.85       |
|   Phase 3.4   |      20000    |      2000     |       0.96       |       0.87       |
|   Phase 3.5   |      20000    |      5000     |       0.95       |       0.88       |
|   Phase 3.6   |      20000    |      10000    |       0.93       |       0.90       |
|   Phase 3.7   |      20000    |      20000    |       0.92       |       0.92       |

> *Note – English F1 decreases slightly as more Turkish data is added, reflecting capacity re-allocation. Scores remain ≥ 0.92.*

---

## Turkish Adaptation Curve
F1 score vs. number of Turkish fine-tuning samples

![Learning Curve](assets/learning_curve.png)

---

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model = AutoModelForTokenClassification.from_pretrained("mehmet0sahinn/xlm-roberta-base-cased-ner-turkish")
tokenizer = AutoTokenizer.from_pretrained("mehmet0sahinn/xlm-roberta-base-cased-ner-turkish")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

text = "Mustafa Kemal Atatürk 1881 yılında Selanik'te doğdu."
ner_results = nlp(text)

for entity in ner_results:
    print(entity)
```

---

## Dataset

- Source: PAN-X from the [google/xtreme](https://huggingface.co/datasets/google/xtreme).
- Languages: English, Turkish
- Training size 20K (EN) + 20K (TR) rows
- Validation size 10K (EN) + 10K (TR)
- Test size 10K (EN) + 10K (TR)

---

## Resources

- [Hugging Face Model](https://huggingface.co/mehmet0sahinn/xlm-roberta-base-cased-ner-turkish)
- [Kaggle Notebook](https://www.kaggle.com/code/mehmet0sahinn/cross-lingual-ner-adaptation)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Demo Locally

```bash
cd gradio
python app.py
```

---

## License

This repository is licensed under the MIT License.
