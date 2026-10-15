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
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import urlopen

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


# ---------------------------------------------------------------------------
# 工具 2：联网 arxiv 查询
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_MAX_RESULTS = 5  # 一次给几条
ABSTRACT_CHARS = 500  # 摘要截到多少字，不能全给，prompt 太大了


@tool
def search_arxiv(query: str) -> str:
    """在 arXiv 上检索论文，返回标题、作者、摘要开头和链接 —— 用来找【本地论文库没有】的论文。

    什么时候用：本地论文库覆盖不到的问题 —— 更新的工作、别的方向，或者 query_corpus 说「没有找到」之后。
    什么时候不该用：问题就在本地那 5 篇（BERT / GPT-2 / GPT-3 / Transformer 等）的范围内时，走 query_corpus；
    闲聊、写代码直接回答即可，两个工具都不用调。
    注意它只给标题和摘要开头，不给全文 —— 要读正文得先下载。

    参数 query 用 arXiv 的查询语法：字段前缀 all:（全文）/ ti:（标题）/ abs:（摘要），
    短语用引号，逻辑用 AND / OR，例如 ti:"chain of thought"、all:diffusion AND abs:survey。
    不写字段前缀时会自动按 all: 处理，所以只给关键词也能用。
    """
    # ① 没带字段前缀就按全文检索
    if ":" not in query:
        query = f"all:{query}"

    # ② 拼 URL
    params = urlencode({"search_query": query, "max_results": ARXIV_MAX_RESULTS})
    url = f"{ARXIV_API}?{params}"

    # ③ 请求 + 兜底 —— 失败必须【返回一句话】，不能抛
    try:
        with urlopen(url, timeout=20) as resp:
            raw = resp.read()
    except Exception as exc:
        return f"arXiv 检索失败（{type(exc).__name__}: {exc}）。可以只依据本地论文"

    # ④ 解析
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return f"arXiv 返回的内容解析不了（{exc}）。可以只依据本地论文库回答。"

    entries = root.findall("a:entry", ARXIV_NS)
    if not entries:
        return f"arXiv 上没有搜到和「{query}」相关的论文。"

    # ⑤ 逐条拼
    pieces = []
    for entry in entries:
        arxiv_id = entry.find("a:id", ARXIV_NS).text.rsplit("/", 1)[-1]
        title = entry.find("a:title", ARXIV_NS).text.strip()
        summary = entry.find("a:summary", ARXIV_NS).text.strip()
        if len(summary) > ABSTRACT_CHARS:
            summary = summary[:ABSTRACT_CHARS] + "…"
        names = [
            a.find("a:name", ARXIV_NS).text for a in entry.findall("a:author", ARXIV_NS)
        ]
        authors = "、".join(names[:3]) + (" 等" if len(names) > 3 else "")
        published = entry.find("a:published", ARXIV_NS).text[:10]
        pieces.append(
            f"[arXiv:{arxiv_id}] {title}\n"
            f"  作者：{authors}（{published}）\n"
            f"  摘要：{summary}"
        )

    total = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    head = f"arXiv 共命中 {total.text} 篇，下面是相关度最高的 {len(entries)} 篇："
    return head + "\n\n" + "\n\n".join(pieces)


# 所有工具的清单 —— 将来加工具，只改这一行
ALL_TOOLS = [query_corpus, search_arxiv]
