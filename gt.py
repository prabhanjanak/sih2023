import streamlit as st
import PyPDF2
import requests
import base64
from transliterate import translit
from transformers import pipeline

API_URL = "https://api-inference.huggingface.co/models/Avanthika/language-translation"
HEADERS = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def translate_to_english(text):
    payload = {"inputs": text}
    response = query(payload)
    translated_text = response.get("outputs", "")
    return translated_text

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

    # Checkbox for language selection
    language_selection = st.checkbox("Document Language: Kannada (Check for Kannada, uncheck for English)")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Display uploaded PDF using a PDF viewer box
        st.header("Uploaded PDF Preview")

        # Encode PDF content to base64
        pdf_content = base64.b64encode(uploaded_file.read()).decode("utf-8")

        # Display the PDF using an iframe
        st.write(f'<iframe src="data:application/pdf;base64,{pdf_content}" width="700" height="500" style="border: none;"></iframe>', unsafe_allow_html=True)

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Translate to English if Kannada is selected
        if language_selection:
            # Transliterate Kannada characters to Roman script
            pdf_text_roman = translit(pdf_text, 'kn', reversed=True)
            pdf_text = translate_to_english(pdf_text_roman)

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
            
            # Correct spelling errors in the user's question
            corrected_question = fix_spelling(user_question, max_length=2048)[0]['generated_text']

            # Answer the corrected question
            answer = answer_question(corrected_question, pdf_text)
            st.write(answer)

if __name__ == "__main__":
    main()
