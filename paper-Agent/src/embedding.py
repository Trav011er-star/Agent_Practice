# 调用 Ollama 里的向量模型
from langchain_ollama import OllamaEmbeddings

# 向量数据库，负责调用向量模型生成向量，并存储到本地数据库中
from langchain_chroma import Chroma


def get_embedding_model():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    return embedding_model


# 按每个 chunk 的内容生成向量，并存储到Chroma数据库中
# vector_store[] =
# vector:
# [0.12, 0.56, 0.91, ...]

# text:
# "Multi-head attention allows..."


# metadata:
# {
#     page: 4,
#     source: "Attention Is All You Need.pdf"
# }
def create_vector_store(documents):
    """
    第一次建库时使用：
    chunks -> embedding -> Chroma
    """
    embedding_model = get_embedding_model()

    # 每次调用 Chroma.from_documents(...)，
    # 是在“把这批 documents/chunks 加入到 Chroma 中”。
    # 重复运行会导致有重复的向量存储数据。
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="vectorstore/chroma_db",
        collection_name="attention_paper",
    )

    return vector_store


def load_vector_store(
    vectorStore_directory="vectorstore/chroma_db",
    vectorStore_conllection_name="attention_paper",
):
    """
    以后查询时使用：
    直接打开已有 Chroma
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=vectorStore_directory,
        # 将查询的自然语言转为向量
        embedding_function=embedding_model,
        collection_name=vectorStore_conllection_name,
    )

    return vector_store
