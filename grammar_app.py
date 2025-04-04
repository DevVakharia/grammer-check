# The file path was changed to grammar_app.py which is in current working directory 
# Previously, the directory `/content/drive/MyDrive/Colab/Notebooks/`
# may not have been mounted or accessible, causing the error. 
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import language_tool_python
import re
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set paths
model_path = '/content/drive/MyDrive/Colab Notebooks/my_grammar_model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models():
    # Load T5 model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    
    # Load LanguageTool
    tool = language_tool_python.LanguageTool('en-US')
    
    return model, tokenizer, tool

def hybrid_grammar_check(text, model, tokenizer, device, tool):
    # First correction with LanguageTool
    corrected_text = tool.correct(text)
    
    # Second correction with T5
    inputs = tokenizer(
        corrected_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text, tool.check(text)

def main():
    st.title("🧑🏫 Grammar Correction Expert")
    model, tokenizer, tool = load_models()
    
    text = st.text_area("Enter your text here:", height=200)
    
    if st.button("Check & Correct"):
        if not text.strip():
            st.error("Please enter some text!")
            return
            
        with st.spinner("🔍 Analyzing grammar..."):
            corrected, matches = hybrid_grammar_check(text, model, tokenizer, device, tool)
            
        st.subheader("Original Text")
        st.write(text)
        
        st.subheader("Corrected Text")
        st.success(corrected)
        
        if matches:
            st.subheader("Found Issues")
            for i, match in enumerate(matches, 1):
                st.markdown(f"""
                **Issue {i}**:
                - *Type*: {match.ruleId}
                - *Message*: {match.message}
                - *Suggestions*: {', '.join(match.replacements[:3])}
                """)
        else:
            st.info("🎉 No grammatical errors found!")

if __name__ == "__main__":
    main()
