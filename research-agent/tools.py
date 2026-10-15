"""工具定义。

一个工具 = 一个普通 Python 函数 + 一段【给模型看】的说明书。

对比 05-ReAct智能体.py 的手搓版：那边把工具描述拼进 prompt 字符串里，
全靠模型自觉按格式输出，我再正则去猜。这边交给 `@tool`：
函数名 → 工具名，docstring → 模型看到的描述，类型注解 → 参数 schema。
"""

from langchain_core.tools import tool
import settings  # noqa: F401  （导入是为了副作用，不是为了用名字）
from agent.engine import retrieve  # noqa: E402 —— 来自 paper-agent
from settings import RETRIEVE_K  # noqa: E402


# ---------------------------------------------------------------------------
# 工具 1：query_corpus —— 在本地论文库里检索
# ---------------------------------------------------------------------------
# bind_tools 干的全部事情，就是往请求体里多塞一个字段:
#   "tools": [
#     {
#       "type": "function",
#       "function": {
#         "name": "query_corpus",
#         "description": "在本地 LLM 论文库里做语义检索，返回最相关的原文片段，每条都带
#   [论文名, p.页码]出处。\n\n什么时候用：...",
#         "parameters": {
#           "properties": { "question": { "type": "string" } },
#           "required": ["question"],
#           "type": "object"
#         }
#       }
#     }
#   ]
# ---------------------------------------------------------------------------


@tool
def query_corpus(question: str) -> str:
    # 这个是给大模型看的工具使用说明，模型调用函数时可以看见
    # 这里算是给 LLM 在解决问题时的提示词， LLM 优先调用工具
    """在本地 LLM 论文库里做语义检索，返回最相关的原文片段，每条都带 [论文名, p.页码] 出处。

    什么时候用：问题涉及这 5 篇论文的方法、结构、实验数字、结论，或它们之间的对比时。
    什么时候不该用：闲聊、写代码、问时效性新闻，或论文库覆盖不到的主题 —— 直接回答即可，不要调用。

    参数 question 请填一个完整的自然语言问句（例如「BERT 和 GPT-3 的预训练目标有什么区别？」），
    不要只写几个关键词 —— 检索器是按语义匹配的，问句越完整，召回越准。
    """
    # 检索并把这批片段拼成一个【字符串】返回。
    docs = retrieve(question, k=RETRIEVE_K)
    #
    # 每个 doc 可用的 metadata：
    #   doc.metadata["paper"] → 论文名（如 "BERT"）
    #   doc.metadata["page"]  → 页码
    #   doc.page_content      → 片段原文
    #
    # 拼的时候【必须】带上出处，形如：[BERT, p.3]
    # 否则模型没法在最终答案里引用，你也就没法验证它是不是编的。
    result = []
    for doc in docs:
        piece = f"[{doc.metadata['paper']}, p.{doc.metadata['page'] + 1}] {doc.page_content}"
        result.append(piece)
    return "\n\n".join(result)


# 所有工具的清单 —— 将来加工具，只改这一行
ALL_TOOLS = [query_corpus]
