import os
import json
import time
import uuid
import logging
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAG:
    def __init__(self):
        self.groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
        self.google_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.embedder = CohereEmbeddings(model="embed-english-light-v3.0")
        self.qdrant_url = os.environ.get("QDRANT_DB_URL")

    def load_documents(self, urls):
        logging.info(f"Loading documents from {urls}")
        web_loader = WebBaseLoader(urls)
        docs = web_loader.load()
        logging.info(f"Loaded {len(docs)} documents.")
        return docs

    def split_documents(self, docs):
        logging.info("Splitting documents into chunks.")
        splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=100)
        chunk_docs = splitter.split_documents(docs)
        logging.info(f"Split documents into {len(chunk_docs)} chunks.")
        return chunk_docs

    def create_qna_index(self, chunk_docs):
        logging.info("Creating QnA index.")
        SYSTEM_PROMPT = """
You are an assistant that generates only answerable questions from a given document chunk.

Task:
Given the text inside <DOCUMENT_CHUNK> tags, list 3 questions that can be answered directly and completely using ONLY the information in that text.

Instructions:
- Consider the chunk as your entire knowledge.
- Generate exactly 3 question-answer pairs.
- The answer for each question must be explicitly stated or clearly implied in the chunk.
- Do NOT include any question that would require outside knowledge to answer.
- Avoid duplicate or trivially rephrased questions.
- Make each question standalone and clear.
- Your output MUST be a single valid JSON object that can be parsed in Python.
Input:
<DOCUMENT_CHUNK>
{chunk}
</DOCUMENT_CHUNK>
"""
        prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        qna_docs = []

        for i, chunk in enumerate(chunk_docs):
            if i != 0 and i % 55 == 0:
                logging.info("Time to Sleep for 60secs..")
                for i in range(60, 0, -1):
                    time.sleep(1)
                    print(f"{i} secs.. ", end="\r", flush=True)

            sys_prmt = prompt.format(chunk=chunk.page_content)
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": sys_prmt + '- The JSON object must conform to this schema: `{"items": [{"Q": "<question>", "A": "<answer>"}]}`.'},
                          {"role": "user",
                           "content": "Generate 3 question-answer pairs based on the document chunk, following all instructions in the system prompt."}],
                response_format={"type": "json_object"},
                model="moonshotai/kimi-k2-instruct-0905")
            
            try:
                queries = json.loads(response.choices[0].message.content)
                for query in queries["items"]:
                    user_query = query["Q"]
                    answer = query["A"]
                    doc = Document(
                        page_content=f"Question: {user_query}\n Answer: {answer}",
                        metadata={"id": str(uuid.uuid4()), "source": chunk.metadata["source"]}
                    )
                    qna_docs.append(doc)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON: {e}")
                logging.error(f"Invalid JSON string: {response.choices[0].message.content}")


        QdrantVectorStore.from_documents(
            documents=qna_docs,
            embedding=self.embedder,
            collection_name="RAG QnA Docs",
            url=self.qdrant_url
        )
        logging.info("QnA vector store created successfully.")

    def create_summary_index(self, docs):
        logging.info("Creating summary index.")
        SYSTEM_PROMPT = "Summarize the following document clearly and concisely. Focus on preserving the main ideas, key details, and overall intent of the text without adding extra information or opinions. Output the summary in 2-3 sentences."
        
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        doc_map = [{"id": doc_ids[i], "doc": doc.page_content} for i, doc in enumerate(docs)]

        with open("doc_map.json", "w") as f:
            json.dump(doc_map, f)

        summary_docs = []
        for i, doc in enumerate(docs):
            user_query = doc.page_content
            response = self.google_client.chat.completions.create(
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
            summary_docs.append(summary_doc)

        QdrantVectorStore.from_documents(
            documents=summary_docs,
            embedding=self.embedder,
            collection_name="RAG Summary Docs",
            url=self.qdrant_url
        )
        logging.info("Summary vector store created successfully.")

    def query_qna_index(self, user_query):
        logging.info(f"Querying QnA index with: {user_query}")
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedder,
            collection_name="RAG QnA Docs",
            url=self.qdrant_url
        )
        results = vector_store.similarity_search(query=user_query, k=3)
        return results

    def query_summary_index(self, user_query):
        logging.info(f"Querying summary index with: {user_query}")
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedder,
            collection_name="RAG Summary Docs",
            url=self.qdrant_url
        )
        results = vector_store.similarity_search(query=user_query, k=3)
        return results

    def get_answer(self, user_query, context):
        logging.info("Getting answer from LLM.")
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
        response = self.google_client.chat.completions.create(
            model="gemini-flash-lite-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )
        return response.choices[0].message.content
