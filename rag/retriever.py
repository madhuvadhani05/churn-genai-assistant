import os
from pathlib import Path

import numpy as np
from google import genai

ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = ROOT / "rag" / "playbook" / "retention_playbook.md"
EMBED_MODEL = "gemini-embedding-001"


def load_chunks():
    text = PLAYBOOK.read_text(encoding="utf-8")
    parts = [p.strip() for p in text.split("## ")[1:]]
    return ["## " + p for p in parts]


def embed(client, texts):
    res = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in res.embeddings])


class Retriever:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.chunks = load_chunks()
        self.vecs = embed(self.client, self.chunks)
        self.vecs /= np.linalg.norm(self.vecs, axis=1, keepdims=True)

    def search(self, query, k=3):
        q = embed(self.client, [query])[0]
        q /= np.linalg.norm(q)
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]