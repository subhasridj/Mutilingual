# Mutilingual
This project is an end-to-end Retrieval-Augmented Generation (RAG) system that enables users to extract accurate answers from large PDF documents (35–400 pages) in real time. It combines hybrid search and LLM-based generation to deliver fast, reliable, and context-aware responses.
❗ Problem Statement

Traditional document search systems fail to provide precise answers from large PDFs due to:

High latency in processing large documents
Poor retrieval accuracy
Inability to handle multiple queries efficiently

This project addresses these challenges by building a scalable and optimized RAG pipeline.

Solution:

The system processes PDFs, retrieves the most relevant content using hybrid search, and generates accurate answers using a language model.

 Pipeline Flow:
PDF ingestion and text extraction
Smart chunking (text + tables) with overlap
Embedding generation
Hybrid retrieval using FAISS + BM25
MMR re-ranking for better context selection
Answer generation using LLM
Response evaluation with scoring system
