import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the question-answering model
tokenizer = AutoTokenizer.from_pretrained("lisa/legal-bert-squad-law")
model = AutoModelForQuestionAnswering.from_pretrained("lisa/legal-bert-squad-law")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    start_positions = model(**inputs).start_logits.argmax(1)
    end_positions = model(**inputs).end_logits.argmax(1)
    
    # Convert token indices to strings
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_positions[0]:end_positions[0]+1])
    
    return answer

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
