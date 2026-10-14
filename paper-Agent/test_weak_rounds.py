"""回归测试：检索「防雷」的两条停止条件。

跑法（在 paper-Agent 目录下）：
    ./venv/Scripts/python.exe test_weak_rounds.py

和 test_api.py 一样是【脚本】，不是 pytest —— 直接跑，看打印。
（本文件会真的调模型、跑图，比较慢，几分钟量级。）

两件事：
  1. 检索失败 —— 图应该很快停下、不报错。
     这一支覆盖的是 top_score = None 那条分支（那行打印前后出过两个 bug：
     一次是三元表达式里的 fresh[0]，一次是 None 的 :.2f）。
  2. 四道回归题 —— 确认「明显该停的题」停下了，同时没把「能答的题」误杀。
     其中最后一道中文题是本次阈值设计的核心依据（见工作日志 10-13）。
"""

import sys

sys.path.insert(0, "agent")

import graph
from graph import graph as g, make_initial_state

# ============================================================
# 1. 检索失败：把 retrieve 换成一个只会抛异常的函数
# ============================================================
# node_collect 里是按【模块名 retrieve】去调的，所以覆盖 graph.retrieve 就生效。
real_retrieve = graph.retrieve


def boom(**kw):
    raise RuntimeError("模拟：检索挂了")


graph.retrieve = boom
print("=" * 24, "1. 检索失败", "=" * 24)
r = g.invoke(make_initial_state("随便问"))
print(
    f">>> hops={r['hops']} weak_rounds={r['weak_rounds']} "
    f"citations={len(r['citations'])}"
)

# ============================================================
# 2. 回归：把真正的 retrieve 换回来，跑四道题
# ============================================================
graph.retrieve = real_retrieve
print("=" * 24, "2. 回归", "=" * 24)

QS = [
    "What is the BLEU score of the big model on WMT 2014 English-to-German?",
    "Compare the pretraining objectives of BERT and GPT-3",
    "BERT 和 GPT-3 的预训练目标有什么不同？",
    "法国的首都是哪里？",
]

for q in QS:
    r = g.invoke(make_initial_state(q))
    print(
        f">>> hops={r['hops']} weak={r['weak_rounds']} "
        f"cit={len(r['citations'])}  << {q}"
    )
