"""检索评测指标。

这里只解决一个问题：**系统检索回来的片段，是不是我们要找的那些？**

两个层次的指标：

1. 页码命中（page hit）
   标准答案记的是"答案在第几页"。只要检索回来的前 k 个片段里，
   有任意一个来自这些页，这题就算命中。

2. 证据命中（evidence hit）—— 更严格
   光落在正确的那一页还不够，还得**真的把那句话捞回来了**。
   判断方法是子串匹配：检索回来的片段里有没有包含我们事先写好的那一段原文。

为什么要有第 2 个指标？
因为这篇论文只有 15 页，每页切成 3~4 个 chunk。
"随便捞 3 个片段就蒙对页码"的概率太高了，只看页码这个指标区分度不够。
证据命中要求真正的原文句子被捞回来，区分度强得多。

注意：这两个指标都**不需要调用任何大模型**，纯字符串比对，跑一遍几秒钟。
"""


def normalize(text):
    """把连续空白（含换行）压成单个空格，再去掉首尾空格。

    因为 PDF 提取出来的文本换行位置很随机，
    直接比字符串会因为换行符对不上而误判。
    """
    return " ".join(str(text).split())


def _page_of(doc):
    """从 Document 的 metadata 里取出页码（PyMuPDFLoader 给的是 0 开始的整数）。"""
    try:
        return int(doc.metadata.get("page", -1))
    except (TypeError, ValueError):
        return -1


def page_hit(docs, gold_pages, k):
    """前 k 个片段里，有没有来自 gold_pages 中任意一页的？有=1，没有=0。"""
    gold = set(int(p) for p in gold_pages)
    for doc in docs[:k]:
        if _page_of(doc) in gold:
            return 1
    return 0


def evidence_hit(docs, evidence, k):
    """前 k 个片段里，有没有真正包含标准答案原文的？有=1，没有=0。"""
    target = normalize(evidence).lower()
    if not target:
        return 0
    for doc in docs[:k]:
        if target in normalize(doc.page_content).lower():
            return 1
    return 0


def reciprocal_rank(docs, gold_pages, cutoff=10):
    """第一个命中的片段排在第几名？第 1 名得 1.0，第 2 名得 0.5，第 3 名得 0.33……

    为什么关心排名：最终只有前 k 个片段会被喂给大模型。
    正确答案排在很后面 = 等于没检索到。所以"排得靠前"本身就有价值。
    MRR = 所有题目的这个分数的平均值。
    """
    gold = set(int(p) for p in gold_pages)
    for rank, doc in enumerate(docs[:cutoff], start=1):
        if _page_of(doc) in gold:
            return 1.0 / rank
    return 0.0


def evaluate_item(docs, item, k_values=(1, 3, 5), mrr_cutoff=10):
    """算单条题目的所有指标，返回一个 dict。"""
    result = {
        "id": item["id"],
        "category": item["category"],
    }
    for k in k_values:
        result[f"page_hit@{k}"] = page_hit(docs, item["gold_pages"], k)
        result[f"evidence_hit@{k}"] = evidence_hit(docs, item["evidence"], k)
    result["reciprocal_rank"] = reciprocal_rank(docs, item["gold_pages"], mrr_cutoff)
    return result


def summarize(per_item_results, k_values=(1, 3, 5)):
    """把每条题目的得分汇总成整体指标。"""
    n = len(per_item_results)
    if n == 0:
        return {}

    summary = {"num_questions": n}
    for k in k_values:
        summary[f"Recall@{k}"] = (
            sum(r[f"page_hit@{k}"] for r in per_item_results) / n
        )
        summary[f"AnswerHit@{k}"] = (
            sum(r[f"evidence_hit@{k}"] for r in per_item_results) / n
        )
    summary["MRR@10"] = sum(r["reciprocal_rank"] for r in per_item_results) / n
    return summary
