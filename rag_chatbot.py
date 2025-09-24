import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from tqdm import tqdm

# ----------------- Configuration -----------------
def load_keys():
    load_dotenv()
    return {
        "gemini_api": os.getenv("GOOGLE_API_KEY"),
        "pinecone_api": os.getenv("PINECONE_API_KEY_3"),
        "index_name": "pdf-3-index"
    }

def init_models(keys):

    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    genai.configure(api_key=keys["gemini_api"])
    llm_model = genai.GenerativeModel("gemini-2.5-pro")

    pc = Pinecone(api_key=keys["pinecone_api"])

    # Create index if it does not exist
    if keys["index_name"] not in pc.list_indexes().names():
        pc.create_index(
            name=keys["index_name"],
            dimension=384,  # default for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(keys["index_name"])
    return embedding_model, llm_model, index

# ----------------- Pinecone Helper -----------------
def is_index_empty(index):
    stats = index.describe_index_stats()
    return stats["total_vector_count"] == 0

def batch_upsert(index, docs_with_vectors, batch_size=1000):
    for i in tqdm(range(0, len(docs_with_vectors), batch_size), desc="Uploading vectors"):
        batch = docs_with_vectors[i:i + batch_size]
        index.upsert(batch)

# ----------------- RAG Logic -----------------
def search_context(index, embedding_model, query, index_name, top_k=10):
    query_vector = embedding_model.embed_query(query)
    response = index.query(
        index=index_name,
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response.get("matches", [])

def build_prompt(context, query):
    with open("prompt_template.txt", "r", encoding="utf-8") as file:
        prompt_template = file.read()
    return prompt_template.format(context=context, query=query)

def generate_answer(llm_model, prompt):
    try:
        gemini_response = llm_model.generate_content(prompt)
        return gemini_response.text.strip()
    except Exception as e:
        print("Gemini generation error:", e)
        return "Sorry, I couldn't generate a valid response."

# ----------------- Streamlit UI -----------------
def init_session(llm_model):
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = []
        st.session_state.chat_model = llm_model

def display_chat_history():
    for entry in st.session_state.chat_session:
        with st.chat_message(entry["role"]):
            st.markdown(entry["text"])

def main():
    global keys
    keys = load_keys()
    embedding_model, llm_model, index = init_models(keys)

    st.set_page_config(page_title="RAG Chatbot", layout="centered")
    st.title("📄 RAG Chatbot - Ask Me About Your PDF")
    
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
        
    # PDF Upload
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Create embeddings
        docs_with_vectors = []
        for i, chunk in enumerate(chunks):
            vector = embedding_model.embed_query(chunk.page_content)
            docs_with_vectors.append({
                "id": f"{uploaded_file.name}_{i}",
                "values": vector,
                "metadata": {"text": chunk.page_content}
            })

        if is_index_empty(index):
            st.info("Index is empty. Uploading vectors...")
            batch_upsert(index, docs_with_vectors)
            st.success("Vectors uploaded successfully!")
        else:
            st.warning("Pinecone Index already contains vectorized data. Skipping upload.")
        
        st.session_state.pdf_processed = True

    # Chat UI
    init_session(llm_model)
    display_chat_history()

    user_input = st.chat_input("Ask your question here...")
    if user_input:
        
        st.session_state.chat_session.append({"role": "user", "text": user_input})
        st.chat_message("user").markdown(user_input)

        matches = search_context(index, embedding_model, user_input, keys["index_name"])
        if matches:
            context = "\n\n".join(match["metadata"]["text"] for match in matches)
            final_prompt = build_prompt(context, user_input)
            bot_answer = generate_answer(st.session_state.chat_model, final_prompt)
        else:
            bot_answer = "No relevant information found in the documents."

        
        st.session_state.chat_session.append({"role": "assistant", "text": bot_answer})
        st.chat_message("assistant").markdown(bot_answer)

if __name__ == "__main__":
    main()
