import json
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# -------------------
# CONFIG
# -------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("kb")


# -------------------
# LOAD DATA
# -------------------
def load_kb():
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------
# INDEX
# -------------------
def index():
    data = load_kb()

    for item in data:
        emb = embedding_model.encode(item["text"]).tolist()

        collection.add(
            ids=[item["id"]],
            documents=[item["text"]],
            embeddings=[emb],
            metadatas=[{"source": item["source"]}]
        )

    print("✅ Indexed")


# -------------------
# RETRIEVE
# -------------------
def retrieve(query, k=3):
    q_emb = embedding_model.encode(query).tolist()

    return collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )


# -------------------
# OLLAMA CALL
# -------------------
def ask_ollama(prompt):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = r.json()

    if "response" not in data:
        print("OLLAMA ERROR:", data)
        return "I don't know"

    return data["response"]


# -------------------
# PROMPT
# -------------------
def build_prompt(question, results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = ""

    for d, m in zip(docs, metas):
        context += f"[Source: {m['source']}]\n{d}\n\n"

    return f"""
You are a strict RAG assistant.

Rules:
- Use ONLY context
- If answer not in context say: I don't know
- Always cite source

Context:
{context}

Question:
{question}
"""


# -------------------
# PIPELINE
# -------------------
def answer(q):
    results = retrieve(q)

    print("\n===================")
    print("QUESTION:", q)

    print("\nSOURCES:")
    for m in results["metadatas"][0]:
        print("-", m["source"])

    prompt = build_prompt(q, results)
    res = ask_ollama(prompt)

    print("\nANSWER:")
    print(res)


# -------------------
# RUN
# -------------------
if __name__ == "__main__":
    index()

    questions = [
        "How long do I have to get a full refund?",
        "How do I reset my password?",
        "What is the stock price today?"
    ]

    for q in questions:
        answer(q)