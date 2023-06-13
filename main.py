import os
import tempfile
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

st.header("Embedding")
with st.sidebar:
    st.header("DB管理")
    db_name = st.text_area("データべース名", "testdb")
    if st.button("DBを作成する"):
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
        pinecone.create_index(db_name, dimension=1536, metric="dotproduct")
    if st.button("DBをリセットする"):
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
        pinecone.delete_index(db_name)

tab1, tab2= st.tabs(["Text", "PDF"])
with tab1:
    input_text = st.text_area("検索したい文章を入力してください")
    st.write("例）https://en.wikipedia.org/wiki/OpenAI")
    if st.button("テキストをエンベッド"):
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
        if input_text:
            with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp:
                tmp.write(input_text)
                tmp_file_name = tmp.name
            loader = TextLoader(tmp_file_name)
            documents = loader.load()
            os.unlink(tmp_file_name)
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            texts = text_splitter.split_documents(documents)
            docsearch =  Pinecone.from_documents(texts, embeddings, index_name=db_name, namespace="pdf")

with tab2:
    input_pdf = st.file_uploader("PDFを選んでください", type=['pdf'])
    if st.button("PDFをエンベッド"):
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
        if input_pdf:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=".pdf") as tmp:
                tmp.write(input_pdf.read())
                tmp_file_name = tmp.name
            loader = PyPDFLoader(tmp_file_name)
            documents = loader.load()
            os.unlink(tmp_file_name)
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            texts = text_splitter.split_documents(documents)
            docsearch =  Pinecone.from_documents(texts, embeddings, index_name=db_name, namespace="pdf")

st.markdown("---")

st.header("Using")
query = st.text_input("検索したい情報を入力してください")
if st.button("検索する"):

    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
    vectorstore = Pinecone.from_existing_index(index_name=db_name, embedding=embeddings, namespace="pdf")
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4"),
        chain_type="stuff",
        vectorstore=vectorstore,
        return_source_documents=True,
        k=5,
    )
    result = qa({"query": query})
    st.write(result)
