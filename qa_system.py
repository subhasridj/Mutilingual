import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class QASystem:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build_faiss_index(self, chunks):
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def answer_query(self, query, top_k=2):
        if self.index is None:
            return "No PDF loaded yet."
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        results = [self.chunks[i] for i in indices[0]]
        return " ".join(results)
