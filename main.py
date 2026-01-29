import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import pypdf
import os
import shutil
import tempfile

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chat with Your KnowledgeBase", layout="wide")
st.title("📚 Chat with Your KnowledgeBase (DeepSeek-R1)")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    accept_multiple_files=False
)

# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file:
    st.success("✅ PDF Uploaded Successfully")

    # Read PDF
    pdf_reader = pypdf.PdfReader(uploaded_file)
    context = "\n\n".join(
        page.extract_text() for page in pdf_reader.pages if page.extract_text()
    )

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(context)

    # Embeddings (LOCAL)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Clean old DB (important for realtime upload)
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    # Create Vector Store
    import tempfile

    temp_dir = tempfile.mkdtemp()

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=temp_dir
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # -----------------------------
    # User Question
    # -----------------------------
    question = st.text_input("Ask a question about the PDF")

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("⚠️ Please enter a valid question")
        else:
            with st.spinner("Thinking..."):
                docs = retriever.get_relevant_documents(question)

                # Prompt Template
                prompt_template = """
                Use ONLY the context below to answer the question.
                If the answer is not found in the context, reply exactly:
                "The answer is not available in the document."

                Context:
                {context}

                Question: {question}

                Detailed Answer:
                """

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=prompt_template
                )

                context_text = "\n\n".join([d.page_content for d in docs])
                final_prompt = prompt.format(
                    context=context_text,
                    question=question
                )

                # LLM (DeepSeek-R1 LOCAL)
                llm = Ollama(
                    model="qwen2.5-coder:7b",
                    temperature=0.2
                )

                result = llm.invoke(final_prompt)

                st.subheader("Answer")
                st.write(result)
