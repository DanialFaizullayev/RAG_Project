from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the saved index (no re-embedding)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = init_chat_model(model="groq:openai/gpt-oss-120b")


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)


prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. You are given context retrieved from the user's document.

Rules:
1. If the context contains the answer, use it first and say it comes from the document.
2. If the context only partly answers the question, give the part from the document, then add what you know from general knowledge, clearly labeled "General knowledge".
3. If the context has nothing relevant, answer from general knowledge and say the document doesn't cover it.
4. Never present general knowledge as if it came from the document. If you are unsure, say so.
5. Start with a one-sentence direct answer, then give details.
6. Copy numbers and currencies exactly as written in the document. Never convert or change the currency.
7. Write in plain text. Do not use markdown, asterisks, or tables.

Context:
{context}

Question: {question}

Answer:""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Ask questions about your document. Type 'exit' to quit.")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue
    print("\n" + chain.invoke(question))