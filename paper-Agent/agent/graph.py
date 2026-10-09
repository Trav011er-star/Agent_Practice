"""
[工位1] ──传送带──> [工位2] ──传送带──> [工位3]

 - 工位 = 干一件事的地方 → 在代码里叫节点（node），本质就是一个函数
 - 传送带 = 决定下一步去哪 → 叫边（edge）
 - 传送带上流动的那个箱子 = 工位之间传递的数据 → 叫 state
"""

# ============================================================
# 第 1 件事：把要用的工具 import 进来
# ============================================================
# StateGraph : 用来"搭图"的模具
# START / END: 两个特殊标记，代表"起点"和"终点"
from langgraph.graph import StateGraph, START, END

# TypedDict : 一种写法，用来描述"这个盒子里有哪些字段"
from typing import TypedDict
from engine import retrieve
import json
from src.llm import get_llm_model
import config
from pathlib import Path


# ============================================================
# 第 2 件事：定义 state（图里流动的那个箱子）
# ============================================================
class State(TypedDict):
    question: str  # 用户的问题（从头到尾不变）
    # （大模型根据之前轮次的问题和已检索结果改写这轮的问题）
    query: str  # 这一轮要用的检索词
    seen: list[str]  # 已经收过的 chunk_id，用来去重
    items: list[str]  # 攒下来的证据【文字】
    hops: int  # 已经跳了几轮
    enough: bool  # 大模型判断是否结束
    answer: str  # 最终检索结果


# ============================================================
# 全局变量
# ============================================================
# 最大跳转轮次
MAX_HOPS = 10

# 规定返回的格式的 json
print("[LLM] 调用 JSON 格式大模型")
_llm_json = get_llm_model(
    provider=config.LLM_PROVIDER,
    format="json",
)  # 需要 LLM 返回 json,判断检索情况
print("[LLM] 调用 TEXT 格式大模型")
_llm_text = get_llm_model(provider=config.LLM_PROVIDER)  # 需要 LLM 说人话,得到检索结果

# ============================================================
# 系统提示词
# ============================================================
PROMPT_JUDGE = """你是一个检索质量评审员。你的工作是判断"目前检索到的证据"够不够回答"用户的问题"。
【用户的问题】
{question}
【目前已经检索到的证据】
{evidence}                         
判断标准（严格一点）：
- 只有当问题所需的**关键数字/事实**都明确出现在证据里，才算够
- 如果只答对了一半（比如问的是英文→德文，却只找到英文→法文的数据），算**不够**                      
只输出一个 JSON 对象。不要解释，不要 markdown 代码块。
{{"enough": true, "next_query": null}}
{{"enough": false, "next_query": "下一轮要用的英文检索词"}}
next_query 的要求：
1. 必须是**英文**（因为论文是英文写的）
2. 要和已有的证据**不重复**，专注于**还缺的那部分信息**
3. 简短，3~8 个词，像搜索引擎的关键词
"""

PROMPT_DONE = """根据下面的【资料】回答【问题】。
【问题】
{question}
【资料】
{evidence}
要求：
1. 只根据【资料】回答，资料里没有的信息**不要编造**
2. 回答里每个关键事实后面要标出处，格式：[论文名, p.页码]
   （资料每段开头的方括号里就是出处，直接抄过来）
3. 如果资料不足以回答，就直接说"资料里没有提到"
"""


# ============================================================
# 第 3 件事：节点
# ============================================================
def node_collect(state: State) -> dict:
    n = state["hops"] + 1
    query = state["query"]
    seen = state["seen"]

    # TODO A：算这轮要取多少条。第 n 轮取 3n 条。
    # 相当于每一轮新增 3 条新信息
    k = 3 * n

    # TODO B：真的去检索。调 retrieve(问题, k=上面那个数)
    docs = retrieve(query=query, k=k)

    # TODO C：过滤。只留下 chunk_id 还没在 seen 里的那些。
    #         每个元素 doc 的 id 在 doc.metadata["chunk_id"]
    fresh = [d for d in docs if d.metadata["chunk_id"] not in seen]

    def _source_label(doc) -> str:
        """给一条证据生成出处标签，例如 [Attention Is All You Need, p.7]"""
        paper = Path(doc.metadata.get("source", "?")).stem  # 去掉路径和 .pdf
        page = doc.metadata.get("page", "?")
        return f"[{paper}, p.{page}]"

    new_texts = [f"{_source_label(d)} {d.page_content}" for d in fresh]

    new_ids = [d.metadata["chunk_id"] for d in fresh]

    print(f"  [collect] 第 {n} 轮：撒网 k={k}，新收 {len(fresh)} 条")

    return {
        "items": state["items"] + new_texts,
        "seen": seen + new_ids,
        "hops": n,
    }

    # 多跳:
    #         原问题              ← 「我要去哪儿」（防止跑偏）
    # 已攒的证据 items        ← 「我已经有什么」（防止重复问）
    # 上一轮的 query          ← 「我刚才是怎么问的」
    #               │
    #               ▼
    #          大模型 想一下
    #               │
    #               ▼
    #    「还缺什么 → 下一轮该问什么」进而构建新一轮更准确的 query


def node_judge(state: State) -> dict:
    """问大模型：证据够了吗？不够的话下一轮问什么？"""
    question = state["question"]
    evidence = "\n\n".join(state["items"])  # 多条证据用空行拼成一整段

    # TODO 1：把模板填好，得到最终要发给大模型的字符串
    # 填入数据
    prompt = PROMPT_JUDGE.format(question=question, evidence=evidence)

    # TODO 2：调用大模型（变量 _llm_json 已经在上面建好了，直接 _llm_json.invoke(...)）
    #         返回的是一个"消息对象"，正文在它的 .content 属性里
    reply = _llm_json.invoke(prompt)

    # TODO 3：reply.content 是一段 JSON 字符串，用 json.loads 解析成字典
    data = json.loads(reply.content)

    # TODO 4：从字典里取两个值
    enough = data["enough"]
    #         防御：如果模型没给出可用的 next_query（None 或空字符串），
    #              就退回用原问题，别让 state["query"] 变成 None 把下一轮搞崩
    next_query = data.get("next_query") or question

    print(f"  [judge] 第 {state['hops']} 轮 → 够了吗：{enough}｜下轮问：{next_query}")

    return {"enough": enough, "query": next_query}


def node_done(state: State) -> dict:
    print("  [done] 够了，可以答了")
    question = state["question"]
    evidence = "\n\n".join(state["items"])  # 多条证据用空行拼成一整段
    prompt = PROMPT_DONE.format(question=question, evidence=evidence)
    answer = _llm_text.invoke(prompt).content
    print(f"检索结果: {answer}")
    return {"answer": answer}


# ============================================================
# 路由函数
# ============================================================
# 规矩：收一个 state，返回一个字符串（也就是"下一步去哪"）。
# 这个函数**不改数据**。
def route_after_judge(state: State) -> str:
    # 根据目前检索的条目数量和轮次决定是否继续或停止
    if state["enough"] or state["hops"] >= MAX_HOPS:
        return "go_done"
    return "go_again"


# ============================================================
# 第 4 件事：把节点连成图
# ============================================================
builder = StateGraph(State)  # 建一个空图，并声明它的 state 用上面那个 State

builder.add_node("collect", node_collect)
builder.add_node("done", node_done)
builder.add_node("judge", node_judge)

builder.add_edge(START, "collect")
builder.add_edge("collect", "judge")

# 条件边
builder.add_conditional_edges(
    # ① 从哪个节点出来
    "judge",
    # ② 谁来决定下一步（路由函数）
    route_after_judge,
    # ③ 映射表：左边 = 路由函数的返回值，右边 = 真实节点名
    #    "这条边可能通向哪几个节点"。
    {
        "go_done": "done",
        "go_again": "collect",
    },
)
# 最后都接入到 END
# builder.add_edge("enough", END)
# builder.add_edge("not_enough", END)
builder.add_edge("done", END)

# compile() 可以理解成"检查图纸、准备开工"。
# 它会检查你连的边有没有指向不存在的节点之类的错误。
graph = builder.compile()

# ============================================================
# 第 5 件事：跑起来
# ============================================================
if __name__ == "__main__":
    q = "What is the BLEU score of the big model on WMT 2014 English-to-German?"
    print("开始运行")
    result = graph.invoke(
        {
            "question": q,
            "query": q,
            "seen": [],
            "items": [],
            "hops": 0,
        }
    )
    print(f"state = {result}")
    print("运行结束")
