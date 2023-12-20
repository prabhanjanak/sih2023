import streamlit as st
import base64
import PyPDF2
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModelForCausalLM
from docx import Document

# Load English spelling correction model
fix_spelling = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load chatbot model
chatbot_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def extract_text_from_pdf(uploaded_file):
    with uploaded_file as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def answer_question(question, context):
    # Correct spelling errors in the user's question
    question = fix_spelling(question, max_length=2048)[0]['generated_text']
    
    # Tokenize the question and context
    input_text = f"Question: {question} Context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate a response using GPT-2 model
    output_ids = fix_spelling.generate(input_ids, max_length=50, num_return_sequences=1)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return answer

def chat_with_model(user_input):
    # Tokenize user input
    input_ids = chatbot_tokenizer.encode(user_input, return_tensors="pt")

    # Generate model response
    model_output = chatbot_model.generate(
        input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7
    )

    # Decode and return the response
    model_response = chatbot_tokenizer.decode(model_output[0], skip_special_tokens=True)
    return model_response

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

    st.title("PDF and Word Text Extractor, Question Answering, and Chatbot with Streamlit")
    st.sidebar.header("Settings")

    # Upload file (either PDF or MS Word)
    uploaded_file = st.file_uploader("Upload a PDF or MS Word file", type=["pdf", "docx"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        if uploaded_file.type == "application/pdf":
            pdf_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            pdf_text = extract_text_from_docx(uploaded_file)

        # Display extracted text
        st.header("Extracted Text")
        st.text(pdf_text)

        # User question input
        user_question = st.text_input("Ask a question about the document:")

        # Answer user's question using the question-answering model
        if user_question:
            st.header("Answer:")
            answer = answer_question(user_question, pdf_text)
            st.write(answer)

        # User chatbot input
        user_input_chatbot = st.text_input("Chat with the Chatbot:")

        # Get response from the chatbot
        if st.button("Get Chatbot Response"):
            if user_input_chatbot:
                model_response = chat_with_model(user_input_chatbot)

                # Display the response
                st.text("Chatbot Response:")
                st.write(model_response)

if __name__ == "__main__":
    main()
