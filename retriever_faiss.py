import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class FAISSRetriever:
    def __init__(self, docs_path="documents", index_path="faiss_index.index",
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_path = docs_path
        self.index_path = index_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []  # store text + metadata
        self.index = None
        self.doc_embeddings = None

        self._load_documents()
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".docs"):
            self._load_index()
        else:
            self.rebuild_index()

    def _load_documents(self):
        self.documents = []
        for filename in os.listdir(self.docs_path):
            file_path = os.path.join(self.docs_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    self.documents.append({
                        "text": text,
                        "source": filename
                    })

    def rebuild_index(self):
        print("⏳ Rebuilding FAISS index...")
        self._load_documents()
        texts = [doc["text"] for doc in self.documents]
        self.doc_embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        embedding_dim = self.doc_embeddings.shape[1]
        
        # Modern ANN index for faster search
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 neighbors
        self.index.hnsw.efConstruction = 200
        self.index.add(self.doc_embeddings)
        
        self._save_index()
        print("✅ FAISS index rebuilt and saved.")

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".docs", "wb") as f:
            pickle.dump(self.documents, f)

    def _load_index(self):
        print("📂 Loading FAISS index and documents...")
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".docs", "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def update_documents(self, incremental=False):
        """Rebuild index; optionally support incremental update."""
        print("⚡ Updating documents...")
        self.rebuild_index()
