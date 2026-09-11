import os
import io
from pathlib import Path

import pymupdf as fitz
from PIL import Image
import pytesseract
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup API keys through the environment.
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Set the GROQ_API_KEY environment variable before running.")

# Set TESSERACT_CMD if tesseract.exe is not available on PATH.
tesseract_cmd = os.getenv("TESSERACT_CMD")
if not tesseract_cmd:
    default_tesseract = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if default_tesseract.exists():
        tesseract_cmd = str(default_tesseract)

if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

pdf_path = r"data\CSR MODULES (1) (2).pdf"

documents = []

with fitz.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf, start=1):
        
        text = page.get_text("text").strip()
        document_type = "native_text"

        
        if len(text) < 20:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image).strip()
            document_type = "ocr_extracted"

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_number,
                        "type": document_type,
                    },
                )
            )


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
processed_splits = text_splitter.split_documents(documents)


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(processed_splits, embedding_model)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


system_prompt = (
    "You are an AI assistant built for document analysis RAG pipelines.\n"
    "Answer the user query dynamically using only the provided context below. "
    "The context includes both direct text extractions and OCR image text maps. "
    "Clean up any evident OCR typos or structural noise inline using context logic.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.1)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chatbot = create_retrieval_chain(retriever, question_answer_chain)


if __name__ == "__main__":
    max_queries = 100
    query_count = 0
    print(
        "Ask questions about the document. "
        "You can ask up to 100 questions; type 'exit' to quit."
    )

    while query_count < max_queries:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a question.")
            continue

        response = rag_chatbot.invoke({"input": user_query})
        print("\n🤖 Chatbot Response:\n", response["answer"])
        query_count += 1

    if query_count == max_queries:
        print("\nYou have reached the 10-question limit. Goodbye!")
