import streamlit as st
import base64
import PyPDF2
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES

def transliterate_kannada_to_roman(text):
    custom_scheme = SchemeMap(SCHEMES[sanscript.KANNADA], SCHEMES[sanscript.HK])
    return custom_scheme.transliterate(text)

def extract_text_from_pdf(uploaded_file):
    with uploaded_file as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

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

    st.title("PDF Text Extractor for Kannada")
    st.sidebar.header("Settings")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Display PDF viewer
        pdf_data = uploaded_file.read()
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        st.write(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" style="border: none;"></iframe>', unsafe_allow_html=True)

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Display button to show extracted text
        if st.button("Show Extracted Text"):
            # Display extracted text
            st.header("Extracted Text from PDF")
            st.text(pdf_text)

        # User question input (you can customize this based on your needs)
        user_question = st.text_input("Ask a question about the PDF:")

        # Perform any Kannada-specific processing or analysis based on user input
        if user_question:
            st.header("Kannada Processing:")
            # You can add your custom processing logic here based on user input
            st.write("Your custom processing result goes here.")

if __name__ == "__main__":
    main()
