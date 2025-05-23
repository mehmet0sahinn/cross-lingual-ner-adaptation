import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and tokenizer
model_name = "mehmet0sahinn/xlm-roberta-base-cased-ner-turkish"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Inference function
def predict_entities(text):
    results = ner_pipeline(text)
    if not results:
        return "No named entities were found."
    output = ""
    for r in results:
        output += f"{r['word']} ({r['entity_group']}): {round(r['score'] * 100, 1)}%\n"
    return output

# Gradio Interface
interface = gr.Interface(
    fn=predict_entities,
    inputs=gr.Textbox(lines=4, placeholder="Enter Turkish text here...", label="Input Text"),
    outputs=gr.Textbox(label="Named Entity Tags"),
    title="Turkish Named Entity Recognition (XLM-RoBERTa)",
    description="This demo performs Named Entity Recognition (NER) on Turkish text using a fine-tuned XLM-R model. It identifies entities such as persons (PER), locations (LOC), and organizations (ORG)."
)

interface.launch()
