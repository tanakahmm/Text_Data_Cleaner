import json
import string
import pandas as pd
from pathlib import Path
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import FakeEmbeddings



def read_file(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        # assumes a column named 'text'
        return df["text"].dropna().tolist()

    else:
        raise ValueError("Unsupported file format")


def chunk_text(text: str, chunk_size: int = 50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def build_outputs(texts: list, source: str, chunk_size: int = 50):
    all_chunks = []
    metadata_dict = {}
    chunk_id = 0

    for doc_id, text in enumerate(texts):
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, chunk_size)

        for chunk in chunks:
            all_chunks.append(chunk)

            metadata_dict[chunk_id] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk,
                "word_count": len(chunk.split()),
                "source": source
            }

            chunk_id += 1

    return all_chunks, metadata_dict

def chunk_text_overlap(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def build_langchain_documents(texts, source, chunk_size=50, overlap=10):
    documents = []
    chunk_id = 0

    for doc_id, text in enumerate(texts):
        cleaned = clean_text(text)
        chunks = chunk_text_overlap(cleaned, chunk_size, overlap)

        for chunk in chunks:
            metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source": source,
                "token_count": token_count(chunk)
            }

            documents.append(
                Document(page_content=chunk, metadata=metadata)
            )

            chunk_id += 1

    return documents


def build_faiss_index(documents):
    embeddings = FakeEmbeddings(size=384)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore



def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    input_file = "data/input.txt"

    texts = read_file(input_file)

    # Step 3 pipeline
    documents = build_langchain_documents(
        texts=texts,
        source=input_file,
        chunk_size=50,
        overlap=10
    )

    vectorstore = build_faiss_index(documents)

    print(f"✅ Built {len(documents)} LangChain Documents")
    print("✅ FAISS index ready for retrieval")

