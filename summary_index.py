import os
import json
import logging
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SYSTEM_PROMPT = "Summarize the following document clearly and concisely. Focus on preserving the main ideas, key details, and overall intent of the text without adding extra information or opinions. Output the summary in 2-3 sentences."

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" )


web_loader =  WebBaseLoader(["https://raw.githubusercontent.com/asinghcsu/AgenticRAG-Survey/refs/heads/main/README.md",
                             "https://docs.cohere.com/page/agentic-rag-mixed-data",
                             "https://docs.cohere.com/docs/routing-queries-to-data-sources",
                             "https://docs.cohere.com/docs/generating-parallel-queries"
                            ])

logging.info("Loading documents...")
docs = web_loader.load()
logging.info(f"Loaded {len(docs)} documents.")

doc_ids = [str(uuid.uuid4()) for _ in docs]
doc_map = [{"id": doc_ids[i], "doc": doc.page_content} for i,doc in enumerate(docs)]

with open("doc_map.json", "w") as f:
    json.dump(doc_map, f)

logging.info("Generating summaries...")

summary_docs = []

for i,doc in enumerate(docs) :
    user_query = doc.page_content
    response = client.chat.completions.create(
            model="gemini-flash-lite-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ], 
            )
    summary = response.choices[0].message.content
    summary_doc = Document(
        page_content=summary,
        metadata={"id": doc_ids[i], "source": docs[i].metadata["source"]}
    )
    logging.info(f"Summary: {summary_doc.page_content}")
    summary_docs.append(summary_doc)


embedder = CohereEmbeddings(model="embed-english-light-v3.0")

vector_storage = QdrantVectorStore.from_documents(
    documents=summary_docs,
    embedding=embedder,
    collection_name = "RAG Summary Docs",
    url=os.environ.get("QDRANT_DB_URL"))
logging.info("Vector store created successfully.")
