"""横向对比不同重排模型。

动机：先用 `BAAI/bge-reranker-base` 做重排，结果指标**反而下降**
（chunk 500：Recall@3 98.2% -> 93.0%，MRR 0.924 -> 0.860）。
而 RRF 融合后的前 5 条已经覆盖 98.2% 的题目，重排只要不乱动就能保底。

所以问题多半出在**重排模型本身**，而不是"重排这个思路没用"。
这个脚本就是为了把这两件事分开：

    python eval/rerank_compare.py
    python eval/rerank_compare.py --model cross-encoder/ms-marco-MiniLM-L-6-v2

会先跑一遍 hybrid 作为参照基线，再对每个重排模型跑 hybrid_rerank。
结果单独存到 eval/results/rerank_compare.json，不覆盖 latest.json。
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.health import OllamaNotReady, ensure_for  # noqa: E402
from src.retrieval import Reranker  # noqa: E402
from eval.run_eval import load_gold_set, print_table, run_one_config  # noqa: E402

DEFAULT_MODELS = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


def main():
    parser = argparse.ArgumentParser(description="对比重排模型")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--model", nargs="+", default=DEFAULT_MODELS,
                        help="要对比的重排模型（HuggingFace 上的名字）")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过 hybrid 基线，只跑重排模型")
    args = parser.parse_args()

    try:
        ensure_for(need_llm=False)
    except OllamaNotReady as exc:
        print("\n[启动自检失败]\n", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)

    gold_set = load_gold_set()
    print(f"评测集：{len(gold_set['items'])} 条问题  chunk_size={args.chunk_size}")

    results = []
    per_details = {}

    # 参照基线：不做重排的 hybrid
    if not args.skip_baseline:
        out = run_one_config(gold_set, args.chunk_size, "hybrid", None)
        if out:
            results.append(out[0])
            per_details["hybrid"] = out[1]

    for model_name in args.model:
        print(f"\n>>> 重排模型：{model_name}")
        reranker = Reranker(model_name=model_name)
        try:
            reranker.model  # 触发加载/下载，把耗时暴露在这里
        except Exception as exc:  # noqa: BLE001 —— 模型名写错/下载失败都归到这里
            print(f"    [跳过] 加载失败：{exc}")
            continue

        out = run_one_config(gold_set, args.chunk_size, "hybrid_rerank", reranker)
        if out is None:
            continue

        result, detail = out
        # 用模型短名区分，否则表里几行都叫 hybrid_rerank
        short = model_name.split("/")[-1]
        result["mode"] = f"rerank:{short}"
        results.append(result)
        per_details[short] = detail

    if not results:
        print("\n没有任何结果。")
        return

    print_table(results)

    config.EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_RESULT_DIR / "rerank_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "per_item": per_details},
            f, ensure_ascii=False, indent=2,
        )
    print(f"明细已保存到 {out_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
