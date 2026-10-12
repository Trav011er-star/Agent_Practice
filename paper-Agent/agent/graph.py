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


# ============================================================
# 防雷：清洗函数
# ============================================================
def _clean_json(raw: str) -> str:
    """把大模型返回的"脏"文本，洗成一段能交给 json.loads 的文本。

    要对付两种脏法：
      1. 被 ```json ... ``` 包起来的
      2. 前后带废话的，比如：好的，我的判断是：{"enough": true, ...}
    """
    # TODO 1：先删掉代码块围栏 —— 把 "```json" 和 "```" 这两个子串去掉。。
    #   大模型爱用 Markdown 代码块来包 JSON。Markdown 里，代码块的写法就是三个反引号开头、三个反引号结尾：
    #   ```json
    #   {"enough": true, "next_query": null}
    #   ```
    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")

    # ┌───────┬──────────┬────────────────────┬─────────────────────┐
    # │       │ 从哪边找  │        返回        │        记忆          │
    # ├───────┼──────────┼────────────────────┼─────────────────────┤
    # │ find  │ 左 → 右   │ 第一个出现的下标    │ 无前缀 = 默认        │
    # ├───────┼──────────┼────────────────────┼─────────────────────┤
    # │ rfind │ 右 → 左   │ 最后一个出现的下标  │ r = right / reverse │
    # └───────┴──────────┴────────────────────┴─────────────────────┘
    # TODO 2：再截出最外层的花括号 —— 只要【第一个 {】到【最后一个 }】之间。
    index1 = raw.find("{")
    index2 = raw.rfind("}")
    # 左闭右开
    raw = raw[index1 : index2 + 1]
    return raw


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
    tried: list[str]
    citations: dict  # 论文中的具体出处


def make_initial_state(question: str) -> State:
    """构造图的初始输入。命令行和服务都从这里取，保证一致。"""
    return {
        "question": question,
        "query": question,  # 第一轮的检索词 = 原问题
        "seen": [],
        "items": [],
        "hops": 0,
        "tried": [],
        "citations": [],
    }


# ============================================================
# 全局变量
# ============================================================
# 最大跳转轮次
MAX_HOPS = 5

# 规定返回的格式的 json
print("[LLM] 调用 JSON 格式大模型")
# 都用 temperature=0 ，方便结果复现和在找不到结果时提前结束
_llm_json = get_llm_model(
    provider=config.LLM_PROVIDER,
    format="json",
    temperature=0,
)  # 需要 LLM 返回 json,判断检索情况
print("[LLM] 调用 TEXT 格式大模型")
_llm_text = get_llm_model(
    provider=config.LLM_PROVIDER,
    temperature=0,
)  # 需要 LLM 说人话,得到检索结果

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
- 如果已经检索了多轮、换过很多检索词，证据里始终找不到答案，
  **也判 enough = true**（表示"可以作答了"，作答内容就是"资料里没有"）。
  不要因为"没找到"就一直判 false。    
【已经问过的检索词】（next_query 绝不能和这些重复）
{tried}              
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
    try:
        docs = retrieve(query=query, k=k)
    except Exception as e:
        print(f"  [collect] [!!!] 检索失败：{type(e).__name__}: {e}")
        docs = []  # ← 降级成"这轮啥也没捞到"

    # TODO C：过滤。只留下 chunk_id 还没在 seen 里的那些。
    #         每个元素 doc 的 id 在 doc.metadata["chunk_id"]
    fresh = [d for d in docs if d.metadata["chunk_id"] not in seen]

    def _source_label(doc) -> str:
        """给一条证据生成出处标签，例如 [Attention Is All You Need, p.7]"""
        paper = doc.metadata.get("paper", "?")  # 建库时已经写好了论文名
        page = doc.metadata.get("page", "?")
        # page 是 0 起算的（PyMuPDF 的规定），显示时 +1 才对得上人翻的页码
        page = page + 1 if isinstance(page, int) else "?"
        return f"[{paper}, p.{page}]"

    new_texts = [f"{_source_label(d)} {d.page_content}" for d in fresh]

    new_ids = [d.metadata["chunk_id"] for d in fresh]

    print(f"  [collect] 第 {n} 轮：撒网 k={k}，新收 {len(fresh)} 条")

    def _source_meta(doc) -> dict:
        """只算数据，不拼字符串。"""
        paper = doc.metadata.get("paper", "?")
        page = doc.metadata.get("page", "?")
        page = page + 1 if isinstance(page, int) else "?"
        chunk_id = doc.metadata["chunk_id"]
        return {
            "paper": paper,
            "page": page,
            "chunk_id": chunk_id,
        }

    citation = [_source_meta(d) for d in fresh]

    return {
        "items": state["items"] + new_texts,  # 用 + 造新 list
        "seen": seen + new_ids,
        "hops": n,
        "tried": state["tried"] + [query],
        "citations": state["citations"] + citation,
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
    tried_text = "\n".join(state["tried"])
    # TODO 1：把模板填好，得到最终要发给大模型的字符串
    # 填入数据
    prompt = PROMPT_JUDGE.format(question=question, evidence=evidence, tried=tried_text)

    # TODO 2：调用大模型（变量 _llm_json 已经在上面建好了，直接 _llm_json.invoke(...)）
    #         返回的是一个"消息对象"，正文在它的 .content 属性里
    reply = _llm_json.invoke(prompt)

    # TODO 3：reply.content 是一段 JSON 字符串，用 json.loads 解析成字典
    #          【为什么整块都包在 try 里】下面三件事都可能炸，缺一个 try 就白写：
    #            ① _clean_json  —— reply.content 万一不是字符串（比如 None）
    #                               None.replace(...) 会抛 AttributeError
    #            ② json.loads    —— 文本不是合法 JSON → JSONDecodeError
    #            ③ data.get(...) —— 模型返回的是【合法 JSON 但不是对象】
    #                               比如 ["enough", true]，data 就成了 list，
    #                               list 没有 .get → AttributeError
    try:
        raw = _clean_json(reply.content)
        data = json.loads(raw)
        raw_enough = data.get("enough", False)
        # 防御：如果模型没给出可用的 next_query（None 或空字符串），
        #      就退回用原问题，别让 state["query"] 变成 None 把下一轮搞崩
        # 如果没给下一个问题，那么 data.get("next_query") = None，使用 question
        next_query = data.get("next_query") or question

        # 归一化：模型可能给布尔 true，也可能给字符串 "true"。
        # ⚠️ 不能直接 if data.get("enough") —— 字符串 "false" 在 Python 里是【真值】，
        #    会把"不够"误判成"够了"。所以统一转成小写字符串再比。
        enough = str(raw_enough).strip().lower() == "true"

    except Exception as e:
        # 【为什么抓 Exception 而不是 JSONDecodeError】
        # 这里是系统边界：输入来自大模型，它什么都干得出来，上面的失败模式已知就有三种。
        # 宽抓是故意的 —— 宁可降级，也不能让整个图崩。
        # 放心：KeyboardInterrupt（Ctrl+C）继承自 BaseException，不会被这里吞掉。
        # ⚠️ 但宽抓的铁律是【必须留痕】：把异常类型也打出来。
        #    否则哪天是你自己笔误（比如 NameError），也会被静默吃掉，然后查三天。
        print(f"  [judge] [!!!] 解析模型返回失败：{type(e).__name__}: {e}")
        print(f"  [judge] 模型原始返回：{reply.content!r}")

        # 降级：连"够不够"都问不出来了，直接收工作答
        # （答案会是"资料里没有提到"，这是诚实的答案，不是错误）
        # 选 True 不选 False：再跳一轮大概率一路空转到 MAX_HOPS，白等五轮
        enough = True
        next_query = question

    print(f"  [judge] 第 {state['hops']} 轮 → 够了吗：{enough}｜下轮问：{next_query}")

    return {"enough": enough, "query": next_query}


def node_done(state: State) -> dict:
    if state["enough"]:
        print("  [done] judge 认为证据够了，开始作答")
    else:
        print(f"  [done] 跳满 {MAX_HOPS} 轮证据仍不足，如实作答")

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
    # 涉及一篇论文，一次就能回答
    q = {
        "one": "What is the BLEU score of the big model on WMT 2014 English-to-German?",
        "can't answer": "What is Hamburger",
        "multi jump": "Compare the pretraining objectives of BERT and GPT-3",
    }

    print("开始运行")
    result = graph.invoke(make_initial_state(q["multi jump"]))
    # print(f"state = {result}")
    print("运行结束")
