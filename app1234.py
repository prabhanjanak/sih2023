import streamlit as st
import PyPDF2
from transformers import pipeline

# Load the English spelling correction model
fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

def extract_text_from_pdf(uploaded_file):
    with uploaded_file as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def answer_question(question, context):
    # Correct spelling errors in the user's question
    question = fix_spelling(question, max_length=2048)[0]['generated_text']
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
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

    st.title("Right Brothers")
    st.sidebar.header("Settings")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)

            # Display button to show extracted text
            if st.button("Show Extracted Text"):
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

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
