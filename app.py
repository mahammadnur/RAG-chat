import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
from docx import Document
import datetime

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("Gemini_API_KEY")
if not api_key:
    st.error("Please set your Gemini_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=api_key)

# Maintain session state for conversations
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def get_text_from_pdf(pdf_files):
    text = ""
    if not pdf_files:
        return text
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF file {pdf_file.name}: {str(e)}")
            continue
    return text.strip()

def chunks_from_txt(text):
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def vector_store_from_chunks(chunks):
    if not chunks:
        raise ValueError("No text chunks to process.")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

def get_conversation_chain():
    prompt_template = """
    Answer the question based on the provided context. If the answer is not in the context, say "I don't know based on the provided documents."

    Context: {context}
    Question: {question}

    Answer:
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        raise

def user_input(user_question):
    try:
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process PDF files first!")
            return
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vector_store.similarity_search(user_question, k=3)
        if not docs:
            st.write("No relevant information found in the uploaded documents.")
            return
        chain = get_conversation_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("**Response:**")
        st.write(response['output_text'])

        # Save Q&A to conversation history
        st.session_state.conversation_history.append((user_question, response['output_text']))

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def download_conversation_as_pdf():
    history = st.session_state.conversation_history
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, (q, a) in enumerate(history, 1):
        pdf.multi_cell(0, 10, f"Q{i}: {q}")
        pdf.multi_cell(0, 10, f"A{i}: {a}")
        pdf.ln()
    filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

def download_conversation_as_doc():
    history = st.session_state.conversation_history
    doc = Document()
    doc.add_heading('Conversation History', 0)
    for i, (q, a) in enumerate(history, 1):
        doc.add_heading(f"Q{i}: {q}", level=2)
        doc.add_paragraph(f"A{i}: {a}")
    filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(filename)
    return filename

def main():
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📄",
        layout="wide"
    )

    st.header("Chat with Multiple PDFs 📄")
    st.markdown("Upload PDF files (e.g. resumes) and ask questions about their content!")

    user_question = st.text_input("Ask a question about the uploaded documents:")

    if user_question:
        user_input(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.title("📁 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files (resumes, reports, etc.)"
        )

        if st.button("Submit & Process", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
            else:
                with st.spinner("Processing PDF files..."):
                    try:
                        raw_text = get_text_from_pdf(pdf_docs)
                        if not raw_text:
                            st.error("No text could be extracted from the PDF files!")
                            return
                        st.info(f"📝 Extracted text length: {len(raw_text)} characters")

                        text_chunks = chunks_from_txt(raw_text)
                        if not text_chunks:
                            st.error("No text chunks could be created!")
                            return
                        st.info(f"📄 Created {len(text_chunks)} text chunks")

                        vector_store_from_chunks(text_chunks)

                        st.success("✅ PDF files processed and indexed successfully!")
                        st.balloons()

                    except Exception as e:
                        st.error(f"Error processing PDF files: {str(e)}")

        if pdf_docs:
            st.subheader("📋 Uploaded Files:")
            for i, pdf in enumerate(pdf_docs, 1):
                st.write(f"{i}. {pdf.name}")

    # Conversation history download section
    if st.session_state.conversation_history:
        st.subheader("💾 Download Conversation History")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Download as PDF"):
                pdf_file = download_conversation_as_pdf()
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF", f, file_name=pdf_file)

        with col2:
            if st.button("📝 Download as DOC"):
                doc_file = download_conversation_as_doc()
                with open(doc_file, "rb") as f:
                    st.download_button("Download DOC", f, file_name=doc_file)

if __name__ == "__main__":
    main()
