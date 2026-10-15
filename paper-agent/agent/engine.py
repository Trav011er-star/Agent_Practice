import sys
from pathlib import Path

# 手动把项目根目录插到搜索路径最前面
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.embedding import load_vector_store
from src.retrieval import HybridRetriever, Reranker

# ---------------------------------------------------------------
# 二、模块级"缓存"
# ---------------------------------------------------------------
# 前面加下划线是命名习惯，意思是"本文件内部用的，外面别动"。
# 一开始都是 None，表示"还没造"。
_retriever = None
_reranker = None


def get_engine():
    """返回 (retriever, reranker)。第一次调用真的去造，之后只是取出来。"""
    # TODO 1：声明我要改的是**模块级**那两个变量。
    #         （一个关键字，写在函数最开头。不写这行，
    #           Python 会以为你在函数里新建两个同名局部变量，
    #           外面那两个永远是 None。）
    global _retriever, _reranker

    # TODO 2：判断"还没造过"（缓存变量还是 None）。
    if _retriever is None:
        print("[engine] 第一次调用，开始造引擎 ...")
        print("[engine] 正在加载向量库 + 建 BM25 索引 ...")
        vector_store = load_vector_store(
            config.collection_name_for(
                config.CHUNK_SIZE,
                corpus="multi",
            )
        )  # ① 打开向量库
        _retriever = HybridRetriever(vector_store)  # ② 用向量库包出检索器
        _reranker = Reranker()  # 这一步只是建了个空壳
        _reranker.model  # 碰一下这个属性，模型才真的加载
        # （这是 retrieval.py 里的懒加载 @property）
        print("[engine] 就绪")
    else:
        print("[engine] 已经就绪")

    return _retriever, _reranker


def retrieve(query, k=None):
    """给一个问题，返回最相关的 k 个片段（List[Document]）。"""
    retriever, reranker = get_engine()
    k = k or config.RETRIEVE_K  # k 没传就用 config 里的默认值
    return retriever.search(
        query,
        k=k,
        mode="hybrid_rerank",
        # 用混合检索：语义（向量）检索 + BM25 关键词检索 + 综合评分
        reranker=reranker,
    )


# ===============================================================
# 三、自测
# ===============================================================
if __name__ == "__main__":
    print("--- 第 1 次 get_engine() ---")
    get_engine()  # 会打印加载过程

    print("\n--- 第 2 次 get_engine() ---")
    get_engine()  # 什么都不打印，秒回

    print("\n--- 真正检索一次 ---")
    question = "What is the BLEU score of the big model on WMT 2014 English-to-German?"
    docs = retrieve(question)

    print(f"\n问题：{question}")
    print(f"命中 {len(docs)} 个片段：\n")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"--- {i}. page={meta.get('page')}  chunk_id={meta.get('chunk_id')} ---")
        print(doc.page_content[:200].replace("\n", " "))
        print()
