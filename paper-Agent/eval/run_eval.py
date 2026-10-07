"""检索评测脚本：跑一遍 gold_qa.json 里的所有问题，输出指标表格。

用法：
    # 对比三种检索策略（最常用）
    python eval/run_eval.py --chunk-size 500 --mode dense hybrid hybrid_rerank

    # 对比三个 chunk_size（只测纯向量）
    python eval/run_eval.py --chunk-size 500 1000 2000 --mode dense

    # 两种维度一起扫
    python eval/run_eval.py --chunk-size 500 1000 --mode dense hybrid

    # 打印每条题目的明细（哪几题没检索到，方便分析）
    python eval/run_eval.py --chunk-size 500 --mode hybrid --detail

这个脚本**不调用任何生成模型**，只测检索。
（hybrid_rerank 会用 cross-encoder 重排，那是判别模型，不是生成模型。）
"""

import argparse
import json
import sys
from pathlib import Path

# 让脚本能 import 到项目根目录下的 config / src
# （因为直接运行 python eval/run_eval.py 时，Python 只会把 eval/ 加进搜索路径）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.embedding import load_vector_store  # noqa: E402
from src.health import OllamaNotReady, ensure_for  # noqa: E402
from src.retrieval import HybridRetriever, Reranker  # noqa: E402
from eval.metrics import evaluate_item, summarize  # noqa: E402

# 支持多种检索策略
MODES = ("dense", "hybrid", "hybrid_rerank")


def load_gold_set(path=None):
    path = path or (config.EVAL_DIR / "gold_qa.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_retriever(chunk_size):
    """建一个 collection 的检索器。打不开就返回 None。"""
    collection_name = config.collection_name_for(chunk_size)
    vector_store = load_vector_store(collection_name=collection_name)
    return HybridRetriever(vector_store)


def run_one_config(gold_set, chunk_size, mode, reranker, detail=False):
    collection_name = config.collection_name_for(chunk_size)

    try:
        retriever = build_retriever(chunk_size)
    except Exception as exc:
        print(f"[跳过] collection '{collection_name}' 打不开：{exc}")
        print(f"       请先运行：python ingest.py --chunk-size {chunk_size} --rebuild")
        return None

    total = len(retriever.all_documents)
    if total == 0:
        print(f"[跳过] collection '{collection_name}' 是空的。")
        return None

    k = config.EVAL_RETRIEVE_K
    per_item = []
    for item in gold_set["items"]:
        docs = retriever.search(
            item["question"], k=k, mode=mode, reranker=reranker
        )
        per_item.append(evaluate_item(docs, item))

    result = summarize(per_item)
    result["chunk_size"] = chunk_size
    result["mode"] = mode
    result["collection"] = collection_name
    result["num_chunks"] = total

    if detail:
        print(f"\n--- chunk_size={chunk_size} mode={mode} 未命中明细 (Recall@3) ---")
        missed = [
            (item, r)
            for item, r in zip(gold_set["items"], per_item)
            if r["page_hit@3"] == 0
        ]
        if not missed:
            print("  全部命中。")
        for item, r in missed:
            print(
                f"  #{item['id']:<3} gold_pages={item['gold_pages']}  "
                f"RR={r['reciprocal_rank']:.2f}  {item['question']}"
            )

    return result, per_item


def print_table(results, k_values=(1, 3, 5)):
    label = "配置"
    header = (
        f"{label:>18} | {'片段数':>6} | "
        + " | ".join(f"{'Recall@' + str(k):>9}" for k in k_values)
        + " | "
        + " | ".join(f"{'AnsHit@' + str(k):>9}" for k in k_values)
        + f" | {'MRR@10':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        name = f"{r['chunk_size']}/{r['mode']}"
        row = f"{name:>18} | {r['num_chunks']:>6} | "
        row += " | ".join(f"{r['Recall@' + str(k)]:>9.1%}" for k in k_values)
        row += " | "
        row += " | ".join(f"{r['AnswerHit@' + str(k)]:>9.1%}" for k in k_values)
        row += f" | {r['MRR@10']:>7.3f}"
        print(row)
    print()
    print("Recall@k   = 前 k 个片段里至少有一个来自正确页面的题目占比")
    print("AnswerHit@k= 前 k 个片段里真的捞到了标准答案原文的题目占比（更严格）")
    print("MRR@10     = 第一个命中片段排名的倒数平均值，越接近 1 说明排得越靠前")


def save_results(results, per_details):
    config.EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVAL_RESULT_DIR / "latest.json"
    payload = {
        "results": results,
        "per_item": {
            f"{r['chunk_size']}_{r['mode']}": per_details[i]
            for i, r in enumerate(results)
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存到 {out.relative_to(config.PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="+",
        default=[500],
        help="要评测的 chunk_size（需先用 ingest.py 建库）",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["dense", "hybrid", "hybrid_rerank"],
        choices=MODES,
        help="检索策略，可以传多个用于对比",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="打印每条未命中的题目",
    )
    args = parser.parse_args()

    # 检索同样要调用 Ollama 的向量模型，先自检
    try:
        ensure_for(need_llm=False)
    except OllamaNotReady as exc:
        print("\n[启动自检失败]\n", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)

    gold_set = load_gold_set()
    print(f"评测集：{len(gold_set['items'])} 条问题  ({gold_set['paper']})")
    print(f"检索前 {config.EVAL_RETRIEVE_K} 个片段")
    print(f"检索策略：{', '.join(args.mode)}")

    # 重排模型只加载一次（加载很慢，要复用）
    reranker = None
    if "hybrid_rerank" in args.mode:
        print(f"\n加载重排模型 {config.RERANK_MODEL}（首次运行需从 HuggingFace 下载）...")
        reranker = Reranker()
        reranker.model  # 触发加载，把耗时和报错暴露在这里而不是循环里
        print("重排模型就绪。")

    results = []
    per_details = []
    for chunk_size in args.chunk_size:
        for mode in args.mode:
            out = run_one_config(
                gold_set, chunk_size, mode, reranker, detail=args.detail
            )
            if out is None:
                continue
            results.append(out[0])
            per_details.append(out[1])

    if not results:
        print("\n没有可评测的向量库，请先运行 ingest.py。")
        return

    print_table(results)
    save_results(results, per_details)


if __name__ == "__main__":
    main()
