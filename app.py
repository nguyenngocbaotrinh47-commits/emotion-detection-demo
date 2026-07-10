import streamlit as st
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Cache the model so it only loads once from Hugging Face
@st.cache_resource
def load_model():
    # Kéo model thẳng từ repo Hugging Face của em
    # NHỚ THAY 'your-username' thành tên tài khoản Hugging Face của em (ví dụ: 'tina1803')
    model_id = "tina1803/emotion-roberta"
    model = RobertaForSequenceClassification.from_pretrained(model_id)
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

emotion_labels = [
    'anger', 'brain dysfunction (forget)', 'emptiness', 
    'hopelessness', 'loneliness', 'sadness', 
    'suicide intent', 'worthlessness'
]

st.title("Multi-label Emotion Detection Demo 🎭")
st.write("Enter an English sentence to see what emotions the AI detects!")

text = st.text_area("Input Text:", placeholder="Example: I feel so lonely and sad...")

if st.button("Predict"):
    if text:
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=150)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        results = {label: float(prob) for label, prob in zip(emotion_labels, probs) if prob >= 0.5}
        
        if results:
            st.success("### Predicted Emotions:")
            for label, prob in results.items():
                st.write(f"- **{label}**: {prob*100:.2f}%")
        else:
            st.warning("No clear emotions detected.")
    else:
        st.error("Please enter some text in the box!")
