from rag_pipeline import RAGPipeline

def main():
    print("Initializing RAG Chatbot...")
    rag = RAGPipeline()
    print("Chatbot ready! Type 'exit' to quit")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            print("Exiting chatbot.")
            break
        try:
            response = rag.get_response(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()