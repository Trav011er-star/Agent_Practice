"""混合检索（BM25 + 向量）与重排。

背景：为什么要做这个
------------------------------------
当前是纯向量检索。向量检索擅长"语义相近"，但**对专有名词很吃亏**。
论文里满是 BLEU、WMT 2014、NIPS、Table 3 这类词，把它们编码成向量后，
和普通词汇的区分度会被抹平——你问 "WMT 2014 EN-DE 的 BLEU 是多少"，
向量检索可能捞回一堆讲"翻译质量"的泛泛段落。

BM25 是关键词匹配（TF-IDF 家族），对这类词几乎一抓一个准。
两者互补，这就是混合检索的动机。

三种策略，逐级增强：
  1. dense          纯向量检索                        （baseline）
  2. hybrid         BM25 + 向量，用 RRF 融合
  3. hybrid_rerank  hybrid 召回后用 cross-encoder 精排

为什么用 RRF 融合而不是"加权求和"？
因为 BM25 的分数（0~十几）和向量距离（0~2）**量纲完全不同**，
直接加权需要先归一化，而归一化又会引入新参数。RRF 只用**排名**、
不用分数，天然规避了这个问题，是工业界最常用的做法。
"""

import math
import re
from collections import Counter

import config

# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------
# 只保留字母和数字。英文论文足够用。
# （中文语料需要换成分词器如 jieba，本项目暂不涉及。）
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 极常见的英文停用词。BM25 里它们本来就会被 IDF 压到很低，
# 提前去掉可以省一点计算，也让代码意图更清楚。
_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were
    be been being this that these those it its we our they their there here
    which who whom what when where why how not no nor so such than then too very
    can could will would shall should may might must do does did done have has had""".split()
)


def tokenize(text):
    """把一段文本切成小写词元，去掉停用词。"""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class BM25:
    """Okapi BM25 检索器（自己实现，不依赖第三方库）。

    打分公式 —— 对 query 里的每个词 t，累加：

                           f(t, d) * (k1 + 1)
        score += IDF(t) * ------------------------------------
                           f(t, d) + k1 * (1 - b + b * |d|/avgdl)

    各符号含义：
      f(t, d)  词 t 在文档 d 里出现的次数
      |d|      文档 d 的长度（词数）
      avgdl    所有文档的平均长度
      k1       词频饱和参数。控制"出现很多次"的收益递减，
               避免一个词狂刷就能把分数顶上去（默认 1.5）
      b        长度归一化强度。0 = 不管长度，1 = 完全按长度归一化（默认 0.75）
               没有它的话，长文档会因为单纯"词多"而占便宜

      IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
               N     = 文档总数
               n(t)  = 含词 t 的文档数
               出现越少的词权重越高 —— 这正是 "BLEU" 这类专有名词
               能压过 "model" 这种高频词的原因，也是 BM25 和向量检索
               互补的地方。
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        """
        documents: List[Document]，每个元素的 page_content 是原文，
                   metadata 里必须有 chunk_id（用来和向量检索的结果对齐）
        """
        self.k1 = k1
        self.b = b

        self.chunk_ids = [d.metadata["chunk_id"] for d in documents]
        self.tokens = [tokenize(d.page_content) for d in documents]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0

        self.n_docs = len(documents)
        # 每个词出现在多少个文档里（document frequency）
        df = Counter()
        for toks in self.tokens:
            df.update(set(toks))

        # 预先算好每个词的 IDF，查询时直接查表
        self.idf = {
            term: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1)
            for term, n in df.items()
        }

    def search(self, query, k):
        """返回 [(chunk_id, score), ...]，按分数从高到低。"""
        query_terms = tokenize(query)

        scored = []
        for i, doc_tokens in enumerate(self.tokens):
            if not doc_tokens:
                continue
            tf = Counter(doc_tokens)
            length_norm = 1 - self.b + self.b * (self.lengths[i] / self.avgdl)

            score = 0.0
            for term in query_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term)
                if idf is None:
                    continue
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * length_norm)

            if score > 0:
                scored.append((self.chunk_ids[i], score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings, rrf_k=None):
    """Reciprocal Rank Fusion：只按排名融合多路检索结果。

        RRF(d) = Σ_i  1 / (rrf_k + rank_i(d))

    rank_i(d) 是文档 d 在第 i 路检索结果里的名次（从 1 开始）。
    rrf_k 是平滑常数（原论文取 60）：它让"第 1 名和第 2 名"的差距
    不至于过大，从而让多路结果有充分的机会互相补充。

    参数 rankings: [[id, id, ...], [id, id, ...]]  每路已按好坏排序
    返回          : [(id, rrf_score), ...] 按分数从高到低
    """
    rrf_k = config.RRF_K if rrf_k is None else rrf_k

    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# 检索流水线
# ---------------------------------------------------------------------------
class HybridRetriever:
    """把向量检索和 BM25 组合起来，对外提供统一的检索接口。

    使用方式：
        r = HybridRetriever(vector_store)      # 建索引（只需一次）
        docs = r.search("query", k=3, mode="hybrid")
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

        # 一次性读出全部片段，用于建 BM25 索引。
        # include 里不放 embeddings，避免把 768 维向量也拖出来。
        snapshot = vector_store.get(include=["documents", "metadatas"])
        self.all_documents = self._rebuild_documents(snapshot)

        # chunk_id -> Document，融合后靠它把 id 还原成原文
        self.doc_by_id = {d.metadata["chunk_id"]: d for d in self.all_documents}

        self.bm25 = BM25(self.all_documents)

    @staticmethod
    def _rebuild_documents(snapshot):
        """把 Chroma 返回的列式数据还原成 List[Document]。"""
        from langchain_core.documents import Document

        documents = []
        for chunk_id, content, metadata in zip(
            snapshot["ids"], snapshot["documents"], snapshot["metadatas"]
        ):
            meta = dict(metadata or {})
            # chunk_id 作为向量检索和 BM25 共用的主键
            meta.setdefault("chunk_id", chunk_id)
            documents.append(Document(page_content=content, metadata=meta))
        return documents

    # -- 各路检索 ----------------------------------------------------------
    def dense_search(self, query, k):
        """纯向量检索，返回 [(chunk_id, 相似度分数), ...]"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            (doc.metadata["chunk_id"], score) for doc, score in results
        ]

    def bm25_search(self, query, k):
        """BM25 关键词检索，返回 [(chunk_id, 分数), ...]"""
        return self.bm25.search(query, k=k)

    # -- 对外接口 ----------------------------------------------------------
    def search(self, query, k, mode="hybrid", reranker=None,
               candidate_k=None):
        """
        mode:
          "dense"         纯向量
          "hybrid"        BM25 + 向量，RRF 融合
          "hybrid_rerank" hybrid 召回 candidate_k 条后，用 cross-encoder 精排到 k 条

        返回 List[Document]，长度 <= k
        """
        candidate_k = candidate_k or config.HYBRID_CANDIDATE_K

        if mode == "dense":
            ranked = self.dense_search(query, k=k)
            return [self.doc_by_id[cid] for cid, _ in ranked[:k] if cid in self.doc_by_id]

        # ---- 取两路候选 ----
        dense_hits = self.dense_search(query, k=candidate_k)
        bm25_hits = self.bm25_search(query, k=candidate_k)

        # ---- RRF 融合（只用排名，不用分数）----
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in dense_hits], [cid for cid, _ in bm25_hits]]
        )

        if mode == "hybrid":
            return [self.doc_by_id[cid] for cid, _ in fused[:k] if cid in self.doc_by_id]

        if mode == "hybrid_rerank":
            if reranker is None:
                raise ValueError("mode='hybrid_rerank' 需要传入 reranker 实例")
            candidates = [self.doc_by_id[cid] for cid, _ in fused[:candidate_k]
                          if cid in self.doc_by_id]
            return reranker.rerank(query, candidates, top_n=k)

        raise ValueError(f"未知的检索模式: {mode!r}")


# ---------------------------------------------------------------------------
# 重排（cross-encoder）
# ---------------------------------------------------------------------------
class Reranker:
    """用 cross-encoder 对候选片段做精细打分。

    和向量检索的区别（面试常问）：
      - 向量检索是 **双塔（bi-encoder）**：query 和文档**各自**编码成向量，
        再算余弦距离。快，但两者从未"见面"，交互信息丢失。
      - cross-encoder 把 **query 和文档拼在一起**送进模型，
        让注意力机制在两者之间充分交互。准，但每条候选都要跑一次前向，
        不能预先建索引，所以**慢**。

    因此标准做法是两段式：向量检索粗排召回 20 条（快）→ cross-encoder
    精排取 3 条（准）。就像招聘：HR 按关键词粗筛简历，面试官再逐份精读。

    这里用 BAAI 的 bge-reranker（中文社区常用、本地可跑）。
    sentence_transformers 采用懒加载，这样没装 torch 时也不影响其它功能。
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or config.RERANK_MODEL
        self._model = None  # 懒加载

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query, documents, top_n):
        """给每个 (query, document) 对打分，返回分数最高的 top_n 个文档。"""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: -float(x[1]))
        return [doc for doc, _ in ranked[:top_n]]
