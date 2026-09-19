from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()

# 2. LOAD ALL PDF DOCUMENTS
documents_folder = Path(
    "F:/Codes/Projects/AI_Data_Analyst/documents"
)

all_documents = []

pdf_files = list(documents_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.")

for pdf_file in pdf_files:

    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    documents = loader.load()

    # Add source filename to metadata
    for document in documents:
        document.metadata["source_file"] = pdf_file.name

    all_documents.extend(documents)

print("\nAll PDFs loaded successfully!")
print("Total pages:", len(all_documents))

# 3. SPLIT DOCUMENTS INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_documents)

print("Number of chunks:", len(chunks))

# 4. CREATE EMBEDDINGS
print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings created successfully!")

# 5. CREATE VECTOR DATABASE
print("Creating vector database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector database created successfully!")

# 6. CREATE RETRIEVER
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)

# 7. CREATE LOCAL LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 8. CREATE PROMPT
prompt = ChatPromptTemplate.from_template(
    """
You are an HR assistant.

Answer the user's question using ONLY the
information provided in the context.

The context may contain information from
multiple HR documents.

Use information from multiple documents when
necessary to answer the question.

If the answer cannot be found in the context,
say:

"I don't have enough information in the
provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 9. RAG FUNCTION
def ask_question(question):

    # 1. RETRIEVE RELEVANT DOCUMENTS
    retrieved_docs = retriever.invoke(question)

    # 2. CREATE CONTEXT FOR LLM
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # 3. CREATE PROMPT
    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # 4. GET ANSWER FROM LLM
    response = llm.invoke(messages)

    # 5. COLLECT SOURCE INFORMATION
    sources = []

    for doc in retrieved_docs:
        source_file = doc.metadata.get(
            "source_file",
            "Unknown document"
        )

        # PyPDFLoader stores page numbers starting at 0.
        # Convert to human-readable page numbers starting at 1.
        page_number = doc.metadata.get(
            "page",
            None
        )

        if page_number is not None:
            page_number = page_number + 1

        source = {
            "file": source_file,
            "page": page_number,
            "content": doc.page_content
        }

        # Avoid duplicate source chunks
        if source not in sources:
            sources.append(source)

    # 6. RETURN ANSWER + SOURCES
    return {
        "answer": response.content,
        "sources": sources
    }

# 10. TEST RAG WHEN RUN DIRECTLY
if __name__ == "__main__":

    question = "How many annual leave days do employees receive?"

    result = ask_question(question)

    print("\n================================")
    print("RAG ANSWER")
    print("================================")

    print(result["answer"])

    print("\n================================")
    print("SOURCES")
    print("================================")

    for source in result["sources"]:

        print(
            f"- {source['file']} | "
            f"Page {source['page']}"
        )