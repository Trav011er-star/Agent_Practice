"""工具定义。

一个工具 = 一个普通 Python 函数 + 一段【给模型看】的说明书。

对比 05-ReAct智能体.py 的手搓版：那边把工具描述拼进 prompt 字符串里，
全靠模型自觉按格式输出，我再正则去猜。这边交给 `@tool`：
函数名 → 工具名，docstring → 模型看到的描述，类型注解 → 参数 schema。
"""

from langchain_core.tools import tool

from agent.engine import retrieve          # ← 来自 paper-agent（靠 settings.py 搭的桥）
from settings import RETRIEVE_K


# ---------------------------------------------------------------------------
# 工具 1：query_corpus —— 在本地论文库里检索
# ---------------------------------------------------------------------------
@tool
def query_corpus(question: str) -> str:
    """TODO(你写)：这段 docstring 就是模型唯一能看到的"说明书"。

    它写得好不好，直接决定模型【调不调】、【参数填得对不对】。
    要交代清楚三件事：
      1. 这个工具能干什么        —— 在本地论文库（5 篇 LLM 论文）里检索原文片段
      2. 什么时候该用它          —— 问论文里的方法 / 数字 / 结论 / 对比时
      3. 什么时候【不该】用它    —— 闲聊、写代码、问最新新闻（库里没有）

    第 3 条最容易被忽略：不写清楚"不该用"的边界，
    模型会在你问"今天天气"时也去翻论文库，然后拿一堆无关片段硬编。
    """
    # TODO(你写)：真正去检索，并把这批片段拼成一个【字符串】返回。
    #
    #   docs = retrieve(question, k=RETRIEVE_K)
    #
    # 每个 doc 可用的 metadata：
    #   doc.metadata["paper"] → 论文名（如 "BERT"）
    #   doc.metadata["page"]  → 页码
    #   doc.page_content      → 片段原文
    #
    # 拼的时候【必须】带上出处，形如：[BERT, p.3]
    # 否则模型没法在最终答案里引用，你也就没法验证它是不是编的。
    #
    # ⚠️ 返回值是给【模型】读的，不是给人读的：
    #    不要 print，不要塞调试信息，别让噪音挤占上下文。
    raise NotImplementedError


# 所有工具的清单 —— 将来加工具，只改这一行
ALL_TOOLS = [query_corpus]
