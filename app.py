import logging
from rag_techniques.rag import RAG
from rag_techniques.utils import convert_json_to_toon

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    rag = RAG()
    
    urls = ["https://raw.githubusercontent.com/asinghcsu/AgenticRAG-Survey/refs/heads/main/README.md",
            "https://docs.cohere.com/page/agentic-rag-mixed-data",
            "https://docs.cohere.com/docs/routing-queries-to-data-sources",
            "https://docs.cohere.com/docs/generating-parallel-queries"]

    while True:
        print("\nSelect an option:")
        print("1. Create Q&A index")
        print("2. Create summary index")
        print("3. Query using Q&A index")
        print("4. Query using summary index")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            docs = rag.load_documents(urls)
            chunk_docs = rag.split_documents(docs)
            rag.create_qna_index(chunk_docs)
        elif choice == "2":
            docs = rag.load_documents(urls)
            rag.create_summary_index(docs)
        elif choice == "3":
            user_query = input("\nEnter your query: ")
            results = rag.query_qna_index(user_query)
            context = ""
            for res in results:
                context += convert_json_to_toon(res.page_content)
                context += "\n" + "-" * 10
            answer = rag.get_answer(user_query, context)
            print("\nAnswer:")
            print(answer)
        elif choice == "4":
            user_query = input("\nEnter your query: ")
            results = rag.query_summary_index(user_query)
            context = ""
            for res in results:
                context += convert_json_to_toon(res.page_content)
                context += "\n" + "-" * 10
            answer = rag.get_answer(user_query, context)
            print("\nAnswer:")
            print(answer)
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
