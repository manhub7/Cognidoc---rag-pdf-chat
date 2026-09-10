import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Personal PDF Chatbot",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    """Load the local Hugging Face embedding model once."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


def process_pdf(uploaded_file):
    """Extract text from the uploaded PDF, split it into chunks,
    and create a FAISS vector store.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No text could be extracted from this PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("The PDF did not produce any text chunks.")

        embeddings = get_embeddings()

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )

        return documents, chunks, vectorstore

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def get_llm():
    """Create the Groq chat model."""
    if "GROQ_API_KEY" not in st.secrets:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to "
            ".streamlit/secrets.toml locally or Streamlit Cloud Secrets."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"],
    )


def format_docs(docs):
    """Format retrieved documents with page information."""
    formatted = []

    for doc in docs:
        page_number = doc.metadata.get("page", None)

        if page_number is not None:
            source = f"Page {page_number + 1}"
        else:
            source = "Unknown page"

        formatted.append(
            f"[{source}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def answer_question(question, vectorstore):
    """Retrieve relevant chunks and ask GPT-OSS 120B to answer."""
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return (
            "I couldn't find relevant information in the uploaded PDF."
        ), []

    context = format_docs(relevant_docs)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that answers questions about an uploaded PDF.

Rules:
1. Use ONLY the provided context to answer.
2. Do not invent facts that are not present in the context.
3. If the answer cannot be found in the context, say:
   "I couldn't find that information in the uploaded PDF."
4. Give a clear and concise answer.
5. When useful, mention the relevant page number.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = get_llm()

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return response.content, relevant_docs


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        # Process only when a new PDF is uploaded.
        if st.session_state.file_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                try:
                    documents, chunks, vectorstore = process_pdf(
                        uploaded_file
                    )

                    st.session_state.documents = documents
                    st.session_state.chunks = chunks
                    st.session_state.vectorstore = vectorstore
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success("PDF processed successfully.")

                except Exception as e:
                    st.error(f"Could not process PDF: {e}")

    if st.session_state.vectorstore is not None:
        st.divider()

        st.subheader("Document information")

        st.write(
            f"**File:** {st.session_state.file_name}"
        )

        st.write(
            f"**Pages:** {len(st.session_state.documents)}"
        )

        st.write(
            f"**Chunks:** {len(st.session_state.chunks)}"
        )

        st.divider()

        if st.button("🗑️ Clear document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.documents = None
            st.session_state.chunks = None
            st.session_state.file_name = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.caption("Embedding model: BAAI/bge-small-en-v1.5")
    st.caption("LLM: openai/gpt-oss-120b via Groq")
    st.caption("Vector DB: FAISS")


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("📚 Personal PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)

if st.session_state.vectorstore is None:
    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )

else:
    # Display previous chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages.
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander("🔎 Retrieved sources"):
                    for i, doc in enumerate(
                        message["sources"], start=1
                    ):
                        page = doc.metadata.get("page", None)

                        if page is not None:
                            page_text = f"Page {page + 1}"
                        else:
                            page_text = "Unknown page"

                        st.markdown(
                            f"**Source {i} — {page_text}**"
                        )

                        st.write(
                            doc.page_content
                        )

                        if i < len(message["sources"]):
                            st.divider()

    # Chat input.
    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:
        # User message.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Assistant message.
        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                try:
                    answer, sources = answer_question(
                        question,
                        st.session_state.vectorstore,
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("🔎 Retrieved sources"):
                            for i, doc in enumerate(
                                sources, start=1
                            ):
                                page = doc.metadata.get(
                                    "page",
                                    None,
                                )

                                if page is not None:
                                    page_text = (
                                        f"Page {page + 1}"
                                    )
                                else:
                                    page_text = "Unknown page"

                                st.markdown(
                                    f"**Source {i} — {page_text}**"
                                )

                                st.write(
                                    doc.page_content
                                )

                                if i < len(sources):
                                    st.divider()

                except Exception as e:
                    answer = f"An error occurred: {e}"
                    sources = []
                    st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
