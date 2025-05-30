# RAG-Based Chatbot with Ollama, Chroma, and LangChain

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers queries based on a PDF document via a command-line interface. It uses **Ollama (Mistral)** for text generation, **Chroma** as the vector database for retrieval, and **LangChain** for RAG orchestration. The project is versioned with **Git**.

## Features
- **Document Retrieval**: Indexes PDF text using Chroma and Hugging Face embeddings (`all-MiniLM-L6-v2`).
- **Contextual Responses**: Generates answers with Mistral via Ollama.
- **CLI**: Command-line interface for queries.

