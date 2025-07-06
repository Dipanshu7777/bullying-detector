import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --- Configuration ---
MODEL_PATH = "BERT_Bullying_Detector_Model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model and Tokenizer (with caching) ---
# st.cache_resource is the modern way to cache models in Streamlit
@st.cache_resource
def load_model():
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

# --- Streamlit App Interface ---
st.title("🛡️ Bullying and Toxicity Detector")
st.write("This tool uses a fine-tuned BERT model to analyze text for harmful content. Enter text below to get a prediction.")

user_input = st.text_area("Enter your text here:", "", height=150)

if st.button("Analyze Text"):
    if model and tokenizer and user_input:
        # --- Prediction Logic ---
        encoding = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
            
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
        label = "Bullying" if prediction.item() == 1 else "Not Bullying"

        # --- Display Results ---
        st.write("### Prediction:")
        if label == "Bullying":
            st.error(f"**{label}** (Confidence: {confidence:.2f})")
        else:
            st.success(f"**{label}** (Confidence: {confidence:.2f})")
    elif not user_input:
        st.warning("Please enter some text to analyze.")