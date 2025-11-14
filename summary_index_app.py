import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from toonify_json import convert_json_to_toon
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

context = ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

result_docs = []
user_query = input("\n Enter your query: ")

embedder =  CohereEmbeddings(model="embed-english-light-v3.0")

summary_index_vectorstorage = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name = "RAG QnA Docs",
    url=os.environ.get("QDRANT_DB_URL"))

logging.info(f"Performing similarity search with query: {user_query}")
results = summary_index_vectorstorage.similarity_search(query=user_query, k=3)

with open("doc_map.json", "r") as f:
    doc_map = json.load(f)

    for res in results:
        logging.info(f"res is : {res.metadata['id']}")
        for doc in doc_map:
            if doc["id"] in res.metadata["id"]:
                result_docs.append(doc)


for res in result_docs:
    context += convert_json_to_toon(res)
    context += "\n"+"-"*10


SYSTEM_PROMPT = f"""
                You are a high-assurance RAG assistant. 
                You MUST NOT provide medical, legal, financial, or safety-critical recommendations unless at least one retrieved document directly supports the recommendation. 
                If such support is absent, refuse and recommend a qualified professional. 
                Cite all supporting documents using (Doc #, ¶X) and include a Sources list.
                Do not hallucinate. Validate facts across multiple sources whenever possible.
                If you don't found any data for relevent answer than say 'I don't have answer for this question'

                CONTEXT : 
                {context}
                """


client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" )

response = client.chat.completions.create(
  model="gemini-flash-lite-latest",
  messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query},
  ], 
)

logging.info(f"Gemini response: {response.choices[0].message.content}")
print(response.choices[0].message.content)