import streamlit as st
import fitz  # PyMuPDF
import PyPDF2
from transformers import pipeline
from googletrans import Translator

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
    qa_pipeline = pipeline("question-answering", model="lisa/legal-bert-squad-law")
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def display_pdf(pdf_bytes):
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        st.image(page.get_pixmap(), caption=f"Page {page_num + 1}", use_column_width=True)

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
        # Display PDF viewer
        display_pdf(uploaded_file.read())

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

            # Ask the user for the target language
            target_language = st.selectbox("Select Target Language", ["en", "es", "fr"])  # Add more languages as needed

            # Translate the answer to the target language
            translated_answer = translate_text(answer, target_language)

            # Display translated answer
            st.header("Translated Answer:")
            st.write(translated_answer)

if __name__ == "__main__":
    main()
