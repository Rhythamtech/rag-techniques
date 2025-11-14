import os
import json
import time
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()


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

client = OpenAI(base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"))


web_loader =  WebBaseLoader(["https://raw.githubusercontent.com/asinghcsu/AgenticRAG-Survey/refs/heads/main/README.md",
                             "https://docs.cohere.com/page/agentic-rag-mixed-data",
                             "https://docs.cohere.com/docs/routing-queries-to-data-sources",
                             "https://docs.cohere.com/docs/generating-parallel-queries"
                             ])
docs = web_loader.load()

print(f"Loaded {len(docs)} documents.")
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=100)

chunk_docs = splitter.split_documents(docs)

for i,chunk in enumerate(chunk_docs):

    if i!=0 and i%55==0:
        print("Time to Sleep for 60secs..\n")
        for i in range(60,0,-1):
            time.sleep(1)
            print(f"{i} secs.. ", end="\r", flush=True)

    sys_prmt = prompt.format(chunk=chunk.page_content)
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": sys_prmt +'- The JSON object must conform to this schema: `{"items": [{"Q": "<question>", "A": "<answer>"}]}`.'},
                  {"role": "user",
                   "content": "Generate 3 question-answer pairs based on the document chunk, following all instructions in the system prompt."}],

        response_format={"type": "json_object"},
        model="moonshotai/kimi-k2-instruct-0905")
    
    queries = json.loads(response.choices[0].message.content)

    for query in queries["items"]:
        user_query = query["Q"]
        answer = query["A"]

        doc = Document(
            page_content= f"Question: {user_query}\n Answer: {answer}",
            metadata={"id": str(uuid.uuid4()), "source": chunk.metadata["source"]}
        )
        qna_docs.append(doc)


embedder = CohereEmbeddings(model="embed-english-light-v3.0")

vector_storage = QdrantVectorStore.from_documents(
    documents=qna_docs,
    embedding=embedder,
    collection_name = "RAG QnA Docs",
    url=os.environ.get("QDRANT_DB_URL"))
print("QnA vector store created successfully.")

    
