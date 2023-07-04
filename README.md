# ss_bootcamp

## ローカル環境で動かす場合

1. 必要なパッケージをインストール
```
pip install -r requirements.txt
```

1.  コメントアウトされている箇所をアンコメントし、下記を削除
```
secrets = st.secrets
embeddings = OpenAIEmbeddings(openai_api_key=secrets["OPENAI_API_KEY"])
if "pinecone_initialized" not in st.session_state:
    pinecone.init(api_key=secrets["PINECONE_API_KEY"], environment=secrets["PINECONE_ENVIRONMENT"])
    st.session_state.pinecone_initialized = True
```

1.  .envファイルにAPIキーを入力

1.  main.pyを実行
```
streamlit run yourpath/main.py
```

※requirements.txtではバージョンを固定すると安定します。