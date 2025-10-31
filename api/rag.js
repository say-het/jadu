export default function handler(req, res) {
  res.send(`

from langchain_community.document_loaders import TextLoader   
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.embeddings import HuggingFaceEmbeddings    
from langchain_chroma import Chroma   
from langchain_google_genai import ChatGoogleGenerativeAI  
from dotenv import load_dotenv
import os

curdir = os.path.dirname(__file__)

filepath = os.path.join(curdir, "external", "a.txt")
pers_dir = os.path.join(curdir, "db4", "chroma_db1")

if not os.path.exists(pers_dir):
    print("Vector store does not exist")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    loader = TextLoader(file_path=filepath)
    doc = loader.load()

    print("Document length:", len(doc))
    print("Content sample:", doc[0].page_content[:200])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
    chunks = text_splitter.split_documents(doc)

    print("Chunks count:", len(chunks))
    if len(chunks) > 0:
        print("First chunk snippet:")
        print(chunks[0].page_content[:300])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=pers_dir)
    print("Vector store created successfully.")

else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=pers_dir, embedding_function=embeddings)
    print("Existing vector store loaded.")

query = "How was Ayodhya?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # Filter results based on similarity score
    search_kwargs={"k": 3, "score_threshold": 0.3}  # Get top 3 docs above 0.3 similarity
)

print(type(retriever))

retrieved_docs = retriever.invoke(query)

print("\nRetrieved Documents:")
for i, rdoc in enumerate(retrieved_docs, 1):
    print(f"Document {i}:")
    print(rdoc.page_content[:300], "...\n")
    print("Source:", rdoc.metadata.get("source", "Unknown"), "\n")

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

context = " ".join([rdoc.page_content for rdoc in retrieved_docs])

response = model.invoke([
    {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer accurately."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
])

print("\nGemini Response:")
print(response.content if hasattr(response, "content") else response)

`);
}
