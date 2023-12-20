import streamlit as st
import PyPDF2
from transformers import pipeline

def extract_text_from_pdf(uploaded_file):
    with uploaded_file as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="lisa/legal-bert-squad-law")
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

def main():
    # Set page background color
    st.markdown(
        """
        <style>
            body {
                background-color: #265073;
                color: #2D9596;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("PDF Text Extractor and Question Answering")
    st.sidebar.header("Settings")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Display extracted text
        st.header("Extracted Text from PDF")
        st.text(pdf_text)

        # User question input
        user_question = st.text_input("Ask a question about the PDF:")

        # Answer user's question using the question-answering model
        if user_question:
            st.header("Answer:")
            answer = answer_question(user_question, pdf_text)
            st.write(answer)

if __name__ == "__main__":
    main()
