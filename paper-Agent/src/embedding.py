"""向量化 + 向量库读写。

职责：
  - 把文本片段（chunk）变成向量
  - 存进本地 Chroma 数据库
  - 以后查询时把库打开
"""

# 调用 Ollama 里的向量模型
from langchain_ollama import OllamaEmbeddings

# 向量数据库，负责调用向量模型生成向量，并存储到本地数据库中
from langchain_chroma import Chroma

import config


def get_embedding_model():
    embedding_model = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        # 显式指定地址，避免被环境变量 OLLAMA_HOST 带偏（详见 config.py 注释）
        base_url=config.OLLAMA_BASE_URL,
    )

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
def create_vector_store(documents, collection_name=None, ids=None):
    """
    第一次建库时使用：chunks -> embedding -> Chroma

    collection_name 不传就用 config 里的默认值。
    不同 chunk_size 的实验请传入不同的 collection_name，否则会互相覆盖。

    ids: 显式指定每个片段的 id（我们用 metadata['chunk_id']）。
         混合检索需要它来对齐"向量检索"和"BM25"两路结果，
         不传的话 Chroma 会自己生成随机 UUID，两路结果就对不上了。
    """
    collection_name = collection_name or config.COLLECTION_NAME

    embedding_model = get_embedding_model()

    # 每次调用 Chroma.from_documents(...)，
    # 是在"把这批 documents/chunks 加入到 Chroma 中"。
    # 重复运行会导致有重复的向量存储数据。
    # —— 想避免重复，请用 ingest.py --rebuild，它会先清空同名集合。
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        ids=ids,
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_name=collection_name,
    )

    return vector_store


def load_vector_store(collection_name=None):
    """
    以后查询时使用：直接打开已有 Chroma
    """
    collection_name = collection_name or config.COLLECTION_NAME

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(config.VECTORSTORE_DIR),
        # 将查询的自然语言转为向量
        embedding_function=embedding_model,
        collection_name=collection_name,
    )

    return vector_store


def delete_collection(collection_name):
    """清空某个集合并返回是否成功。用于 --rebuild，解决重复写入的问题。"""
    try:
        vector_store = Chroma(
            persist_directory=str(config.VECTORSTORE_DIR),
            embedding_function=get_embedding_model(),
            collection_name=collection_name,
        )
        vector_store.delete_collection()
        return True
    except Exception:
        # 集合本来就不存在，无需删除
        return False


def count_chunks(collection_name):
    """返回某个集合里有多少个向量（用来确认建库成功）。"""
    vector_store = Chroma(
        persist_directory=str(config.VECTORSTORE_DIR),
        embedding_function=get_embedding_model(),
        collection_name=collection_name,
    )
    return len(vector_store.get()["ids"])
