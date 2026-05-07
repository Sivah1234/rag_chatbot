import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# -----------------------------
# 1️⃣ Load Documents
# -----------------------------
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


def load_youtube(url: str, save_dir: str = "docs/youtube/"):
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        FasterWhisperParser()
    )
    return loader.load()


# -----------------------------
# 2️⃣ Split Documents
# -----------------------------
def split_documents(documents: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# -----------------------------
# 3️⃣ Create Vector DB
# -----------------------------
def create_vectorstore(docs, persist_dir="db/chroma"):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectordb


# -----------------------------
# 4️⃣ Initialize LLM
# -----------------------------
def get_llm():
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("Set GROQ_API_KEY as environment variable")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
    )


# -----------------------------
# 5️⃣ RAG QA Function
# -----------------------------
def ask_question(vectordb, llm, question: str):

    docs = vectordb.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you don't know, say you don't know. "
        "Keep the answer concise (max 3 sentences)."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context + "\n\nQuestion: " + question)
    ]

    response = llm.invoke(messages)
    return response.content


# -----------------------------
# 6️⃣ Main Execution
# -----------------------------
if __name__ == "__main__":

    # ---- Load Data ----
    pdf_docs = load_pdf("embedded-system-design-marwedel.pdf")

    youtube_url = "https://www.youtube.com/watch?v=uFhDGagZzjs"
    yt_docs = load_youtube(youtube_url)

    combined_docs = pdf_docs + yt_docs

    # ---- Split ----
    chunked_docs = split_documents(combined_docs)

    # ---- Vector DB ----
    vectordb = create_vectorstore(chunked_docs)

    # ---- LLM ----
    llm = get_llm()

    # ---- Ask ----
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = ask_question(vectordb, llm, query)
        print("\nAnswer:\n", answer)