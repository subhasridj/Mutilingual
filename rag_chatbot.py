import os
import sys
from pdf_processor import extract_pdf_to_chunks
from retriever_faiss import FAISSRetriever
from gpt4all import GPT4All
from config import DOCUMENTS_DIR, MODEL_PATH

def run_chatbot(pdf_path):
    print(f"📚 Processing PDF: {pdf_path}")
    chunks = extract_pdf_to_chunks(pdf_path, DOCUMENTS_DIR)  # sentence-aware chunks
    
    print("🤖 Initializing model and retriever...")
    model = GPT4All(MODEL_PATH, allow_download=False)
    retriever = FAISSRetriever(docs_path=DOCUMENTS_DIR)

    def get_rag_response(prompt):
        # Retrieve top 3 documents with MMR-style ranking (if retriever supports)
        retrieved_docs = retriever.retrieve(prompt, top_k=3)
        context = "\n--- Retrieved Document ---\n".join(retrieved_docs)
        
        full_prompt = (
            "You are a helpful AI assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\nAnswer:"
        )
        with model.chat_session() as session:
            return session.generate(prompt=full_prompt)
    
    print("\n💬 Text Chat Mode. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        print("Bot:", get_rag_response(user_input))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_chatbot.py /path/to/your/file.pdf")
        exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        exit(1)

    run_chatbot(pdf_path)
