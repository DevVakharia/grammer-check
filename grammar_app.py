# app.py
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import language_tool_python
import re

# Cache resources to improve performance
@st.cache_resource
def load_resources():
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained('./my_grammar_model').to(device)
    tokenizer = T5Tokenizer.from_pretrained('./my_grammar_model')
    
    # Initialize LanguageTool with persistence
    tool = language_tool_python.LanguageTool('en-US')
    
    return model, tokenizer, device, tool

def hybrid_correction(text, model, tokenizer, device, tool):
    try:
        # First pass with LanguageTool
        matches = tool.check(text)
        corrected = tool.correct(text)
        
        # Second pass with T5 model
        inputs = tokenizer.encode_plus(
            corrected,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
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
        
        return final_text, matches
    
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return text, []

# Streamlit UI
def main():
    st.title("Advanced Grammar Checker")
    st.markdown("""
    **Hybrid grammar checker combining:**
    - T5 Transformer model
    - LanguageTool rule-based corrections
    """)
    
    text_input = st.text_area("Enter text to check:", height=150,
                             placeholder="Paste your text here...")
    
    if st.button("Check Grammar"):
        if not text_input.strip():
            st.warning("Please enter some text to check")
        else:
            with st.spinner("Analyzing text..."):
                model, tokenizer, device, tool = load_resources()
                corrected_text, matches = hybrid_correction(
                    text_input, model, tokenizer, device, tool
                )
                
                # Display results
                st.subheader("Results:")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Original Text:**")
                    st.write(text_input)
                    
                    st.markdown("**Corrected Text:**")
                    st.success(corrected_text)
                
                with col2:
                    if matches:
                        st.markdown("**Detected Issues:**")
                        for i, match in enumerate(matches, 1):
                            st.markdown(f"""
                            **{i}. {match.ruleId}**  
                            {match.message}  
                            *Suggested fix:* `{', '.join(match.replacements)}`  
                            """)
                    else:
                        st.info("No grammar issues detected!")

if __name__ == "__main__":
    main()
