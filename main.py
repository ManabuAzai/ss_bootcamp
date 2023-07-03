import os
import tempfile
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone

# for using local making .env file and replace below code
# from dotenv import load_dotenv
# load_dotenv()
# embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]) 
# if "pinecone_initialized" not in st.session_state:
#     pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
#     st.session_state.pinecone_initialized = True

secrets = st.secrets
embeddings = OpenAIEmbeddings(openai_api_key=secrets["OPENAI_API_KEY"])
if "pinecone_initialized" not in st.session_state:
    pinecone.init(api_key=secrets["PINECONE_API_KEY"], environment=secrets["PINECONE_ENVIRONMENT"])
    st.session_state.pinecone_initialized = True

# for making vectorstore
# with st.sidebar:
#     st.header("DB管理")
#     db_name = st.text_area("データべース名", "testdb")
#     if st.button("DBを作成する"):
#         pinecone.create_index(db_name, dimension=1536, metric="dotproduct")
#     if st.button("DBをリセットする"):
#         pinecone.delete_index(db_name)

with st.sidebar:
    st.header("Emdedding")

    input_text = st.text_area("テキストを入力してください")
    st.write("例）https://en.wikipedia.org/wiki/OpenAI")
    if st.button("テキストをエンベッド"):
        if input_text:
            with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp:
                tmp.write(input_text)
                tmp_file_name = tmp.name
            loader = TextLoader(tmp_file_name)
            documents = loader.load()
            os.unlink(tmp_file_name)
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            texts = text_splitter.split_documents(documents)
            docsearch =  Pinecone.from_documents(texts, embeddings, index_name="testdb", namespace="ss_bootcamp")

    st.markdown("---")

    input_pdf = st.file_uploader("PDFを選んでください", type=['pdf'])
    if st.button("PDFをエンベッド"):
        if input_pdf:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=".pdf") as tmp:
                tmp.write(input_pdf.read())
                tmp_file_name = tmp.name
            loader = PyPDFLoader(tmp_file_name)
            documents = loader.load()
            os.unlink(tmp_file_name)
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            texts = text_splitter.split_documents(documents)
            docsearch =  Pinecone.from_documents(texts, embeddings, index_name="testdb", namespace="ss_bootcamp")

st.header("Using")
query = st.text_input("検索したい情報を入力してください")
if st.button("検索する"):
    vectorstore = Pinecone.from_existing_index(index_name="testdb", embedding=embeddings, namespace="ss_bootcamp")
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4"),
        chain_type="stuff",
        vectorstore=vectorstore,
        return_source_documents=True,
        k=5,
    )
    result = qa({"query": query})
    st.subheader("Result")
    st.write(result["result"])
    st.subheader("Source Documents")
    for doc in result["source_documents"]:
        st.write(doc)
