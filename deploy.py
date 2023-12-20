import streamlit as st
import PyPDF2
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

def extract_text_from_pdf(uploaded_file):
    with uploaded_file as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def answer_question(question, context):
    tokenizer = AutoTokenizer.from_pretrained("lisa/legal-bert-squad-law")
    model = AutoModelForQuestionAnswering.from_pretrained("lisa/legal-bert-squad-law")
    
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

def main():
    st.title("PDF Text Extractor and Question Answering")
    st.sidebar.header("Settings")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)

        if st.button("Show Extracted Text"):
            st.header("Extracted Text from PDF")
            st.text(pdf_text)

        user_question = st.text_input("Ask a question about the PDF:")

        if user_question:
            st.header("Answer:")
            answer = answer_question(user_question, pdf_text)
            st.write(answer)

if __name__ == "__main__":
    main()
