"""工具定义。

一个工具 = 一个普通 Python 函数 + 一段【给模型看】的说明书。

对比 05-ReAct智能体.py 的手搓版：那边把工具描述拼进 prompt 字符串里，
全靠模型自觉按格式输出，我再正则去猜。这边交给 `@tool`：
函数名 → 工具名，docstring → 模型看到的描述，类型注解 → 参数 schema。
"""

import re

from langchain_core.tools import tool

import settings  # noqa: F401  （导入是为了副作用：把 paper-agent 挂到 sys.path 上）
from agent import engine as paper_engine  # noqa: E402 —— 来自 paper-agent
from src.chunking import assign_chunk_ids  # noqa: E402
from src.embedding import count_chunks, create_vector_store  # noqa: E402
from src.health import OllamaNotReady, ensure_for  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.splitter import split_documents  # noqa: E402
from settings import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOWNLOADED_DIR,
    PAPERS_COLLECTION,
    PAPERS_DIR,
    RETRIEVE_K,
)

import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
    docs = paper_engine.retrieve(question, k=RETRIEVE_K)
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

    什么时候用：本地论文库检索回来的片段明显答不上这个问题时 —— 比如问题涉及库里没有的论文、更新的工作，或者 query_corpus 返回的片段明显不相关。
    什么时候不该用：问题就在本地那 5 篇（BERT / GPT-2 / GPT-3 / Transformer 等）的范围内时，走 query_corpus。本地库里有原文的，就别去外面找二手材料。
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


# ---------------------------------------------------------------------------
# 工具 3：download_paper —— 把 arXiv 论文下回来、切分、写进本地库
# ---------------------------------------------------------------------------
# ★ 这是本项目第一个【写工具】。
#
# 前两个工具看一眼就走，不留痕迹。这个不一样：它往磁盘写文件、往向量库写数据。
# 写工具必须考虑读工具完全不用想的三件事：
#
#   1. 键稳不稳 —— chunk_id 一旦撞车会【静默覆盖】别人的片段
#      （实测：Chroma 对重复 id 是覆盖不是追加，条数不变、内容被换掉、不报错）
#   2. 缓存失效 —— 库变了，可 engine 里那份 BM25 快照没变，
#      新片段对检索【完全不可见】，而且不报错
#   3. 中途失败 —— 嵌入写到一半挂了，库里会留半篇论文
#
# 1 和 3 靠同一招解决：编号只由论文自己决定（见 src/chunking.py）
#   → 同一篇论文重复入库 = id 完全相同 = 覆盖 = 【幂等】
#   → 所以失败了直接重跑一次就行，不用手工清数据库。
# 幂等算是"穷人版的事务"：不保证原子，但保证【可重试】。

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_BAD_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
ARXIV_PDF_URL = "https://arxiv.org/pdf/"
PAPER_MAX_BYTES = 50 * 1024 * 1024  # 单篇 PDF 上限，纯防呆
PAPER_MAX_STEM_CHARS = 120  # 文件名上限，防 Windows 长路径


def _normalize_arxiv_id(raw):
    """把模型给的 id 清成裸 id；清不出来返回 None。

    模型可能塞进来的形态：2305.10601 / 2305.10601v2 / arXiv:2305.10601 /
    https://arxiv.org/abs/2305.10601 —— 这些都得能认。

    ⚠️ 实测：arXiv 的 id_list 参数【不接受】"arXiv:" 前缀，而且不报错，
    直接静默返回 0 条 —— 和"这个 id 根本不存在"长得一模一样。
    所以格式必须我们自己清干净，指望 API 报错来发现问题是指望不上的。

    顺便去掉版本号：我们要的是"这篇论文"，不是"这个版本"。
    否则同一篇论文下了 v1 和 v2，会在语料库里变成两份。
    """
    match = ARXIV_ID_RE.search(raw or "")
    return match.group(1) if match else None


def _fetch_arxiv_meta(bare_id):
    """按 id 查 arXiv，返回 (标题, 年份)；查不到或解析不了，一律返回 None。"""
    url = f"{ARXIV_API}?{urlencode({'id_list': bare_id})}"
    with urlopen(url, timeout=20) as resp:
        root = ET.fromstring(resp.read())
    entry = root.find("a:entry", ARXIV_NS)
    if entry is None:
        return None
    title_el = entry.find("a:title", ARXIV_NS)
    if title_el is None or not title_el.text:
        return None
    # Atom 会把标题折成多行，压平成一个空格分隔的字符串
    title = " ".join(title_el.text.split())
    published = entry.find("a:published", ARXIV_NS)
    year = published.text[:4] if published is not None and published.text else "?"
    return title, year


def _find_local_duplicate(title):
    """本地语料里是不是已经有这篇了？有就返回文件名，没有返回 None。

    做法很粗：把标题压成"纯小写字母数字"取前 30 位，看它是不是某个本地文件名的子串。
    能挡住的是最常见的那种情况 —— 比如有人让下载 Attention Is All You Need，
    而它本来就在基线那 5 篇里。

    ⚠️ 挡不住"本地文件用了简称"的情况：arXiv 标题是
       "BERT: Pre-training of Deep Bidirectional Transformers..."，
       本地文件却叫 "BERT (2018).pdf"，压完是 "bert2018"，匹配不上。
    知道有这个洞就行，别指望它万能 —— 它能省一次白下载，但不能当幂等保证。
    """
    key = re.sub(r"[^a-z0-9]", "", title.lower())[:30]
    if not key:
        return None
    for path in sorted(PAPERS_DIR.glob("*.pdf")) + sorted(DOWNLOADED_DIR.glob("*.pdf")):
        if key in re.sub(r"[^a-z0-9]", "", path.stem.lower()):
            return path.name
    return None


def _safe_pdf_stem(title, year):
    """标题 -> 能当文件名的形式，例如 'Tree of Thoughts - ... (2023)'。

    实测坑：arXiv 的标题里【带冒号】——
        "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    冒号在 Windows 文件名里是非法字符，不处理会直接 OSError。
    顺便压掉换行和多余空格，再限个长度。
    """
    name = " ".join(title.split())
    name = _BAD_FS_CHARS.sub("-", name).strip(" .-")
    if len(name) > PAPER_MAX_STEM_CHARS:
        name = name[:PAPER_MAX_STEM_CHARS].rstrip(" .-")
    return f"{name} ({year})"


@tool
def download_paper(arxiv_id: str) -> str:
    """把一篇 arXiv 论文下载、切分、写进本地论文库；之后就能用 query_corpus 检索它的正文。

    ⚠️ 这个工具代价【很大】，用之前先想清楚：
      · 要联网下载 PDF，再切分、向量化，通常几十秒，论文长的话几分钟
      · 它会【永久改变本地论文库】—— 加进去的论文不会自动消失

    所以只在下面两种情况用：
      · 用户明确要求「把某篇论文读进来 / 下载下来 / 加进库」
      · search_arxiv 已经找到那篇论文，而且【必须读正文】才能回答
        （search_arxiv 只给标题和摘要开头；摘要够用就别下）
    不要为了"多收集点资料"就下载 —— 能靠本地库的 5 篇或摘要回答的，都不要下正文。

    参数 arxiv_id 只填【编号】，不要填 URL、不要填标题，例如 2305.10601。
    编号从 search_arxiv 的输出里原样抄，格式是 [arXiv:2305.10601]。
    带不带版本号都行（2305.10601 和 2305.10601v2 等价，都会指向最新版）。
    """
    # ① 清 id
    bare = _normalize_arxiv_id(arxiv_id)
    if bare is None:
        return (
            f"「{arxiv_id}」看起来不是 arXiv 编号。请把 search_arxiv 输出里的编号"
            f"原样传进来，形如 2305.10601。"
        )

    # ② 问 arXiv 要标题和年份。
    #    文件名必须【我们】定，不能让模型给 —— 模型拼错一个字母，
    #    同一篇论文就会在库里变成两份（而它自己还不知道）。
    try:
        meta = _fetch_arxiv_meta(bare)
    except Exception as exc:
        return f"查 arXiv 元数据失败（{type(exc).__name__}: {exc}）。可以只用本地已有的论文回答。"
    if meta is None:
        return (
            f"arXiv 上查不到编号 {bare} 这篇论文。"
            f"请确认编号是从 search_arxiv 结果里抄的，不要自己拼。"
        )
    title, year = meta

    # ③ 已经在本地了就别白下 —— 它能省一次下载，但不保证覆盖所有情况（见函数注释）
    duplicate = _find_local_duplicate(title)
    if duplicate:
        return (
            f"本地论文库里已经有这篇了：{duplicate}。不用下载 —— "
            f"直接 query_corpus 检索它的正文就行。"
        )

    # ④ Ollama 没起就早点报，别等下载完、切好、要嵌入的时候才炸
    try:
        ensure_for(need_llm=False)
    except OllamaNotReady as exc:
        return f"Ollama 没准备好，暂时没法把论文入库：\n{exc}"

    # ⑤ 下载。先写 .part 临时文件，下完整了再改名 —— 目录里出现的永远是完整文件。
    #    ⚠️ 半截的 PDF 会被 collect_pdf_paths 当正常语料收进全量 ingest。
    stem = _safe_pdf_stem(title, year)
    DOWNLOADED_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOADED_DIR / f"{stem}.pdf"
    part = dest.with_suffix(".pdf.part")
    overwriting = dest.exists()

    try:
        request = Request(
            f"{ARXIV_PDF_URL}{bare}",
            headers={"User-Agent": "research-agent/0.1 (learning project)"},
        )
        with urlopen(request, timeout=120) as resp, open(part, "wb") as fh:
            total = 0
            while True:
                block = resp.read(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > PAPER_MAX_BYTES:
                    raise ValueError(f"文件已超过 {PAPER_MAX_BYTES // 1024 // 1024}MB，放弃")
                fh.write(block)
    except Exception as exc:
        part.unlink(missing_ok=True)  # 别把半截 PDF 留在语料目录里
        return (
            f"下载 arXiv:{bare} 的 PDF 失败（{type(exc).__name__}: {exc}）。"
            f"可以只用本地已有的论文回答。"
        )
    part.replace(dest)

    # ⑥ 切分 + 打编号。
    #    参数统统从 settings/config 取，绝不在这里写死 500 ——
    #    写死的话，哪天 chunk_size 一改，切出来的片段就和目标集合对不上了。
    try:
        documents = load_pdf(str(dest))
        chunks = split_documents(
            documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    except Exception as exc:
        return f"PDF 下下来了（{dest}），但读取/切分失败（{type(exc).__name__}: {exc}）。"
    if not chunks:
        return f"《{title}》里没切出任何文字（多半是扫描版 PDF，没有文字层），没法入库。"

    assign_chunk_ids(chunks, CHUNK_SIZE)
    for chunk in chunks:
        # arXiv 编号记进 metadata —— 文件名里放不下它（会被标题占满），
        # 但将来要追溯"这条片段是哪篇论文的哪一版"就得靠它。
        chunk.metadata["arxiv_id"] = bare

    # ⑦ 写进向量库（复用 paper-agent 那份带批处理的实现，不另写一份）
    try:
        before = count_chunks(PAPERS_COLLECTION)
    except Exception:
        before = 0  # 集合还不存在
    try:
        create_vector_store(
            chunks,
            collection_name=PAPERS_COLLECTION,
            ids=[c.metadata["chunk_id"] for c in chunks],
        )
    except Exception as exc:
        return (
            f"下载和切分都成功了（{len(chunks)} 个片段），但写向量库时报错："
            f"{type(exc).__name__}: {exc}\n"
            f"⚠️ 库现在可能是【半成品】（批量写入写到一半挂的）。"
            f"直接重跑一次 download_paper(arxiv_id=\"{bare}\") 就行 —— "
            f"编号只由论文自己决定，重跑会用同样的 id 覆盖回去，不用手工清库。"
        )
    after = count_chunks(PAPERS_COLLECTION)

    # ⑧ ★ 库变了，但【当前进程里那份检索器还是旧的】：
    #    BM25 是建引擎时抓的一次性快照，doc_by_id 是同一份快照 ——
    #    不重置的话，刚下载的论文对检索完全不可见，而且全程不报错。
    #    放在工具里调用，而不是交给调用方，因为"谁改了库，谁负责让缓存失效"。
    paper_engine.reset_engine()

    return (
        f"已{'覆盖更新' if overwriting else '下载并入库'}：《{title}》（arXiv:{bare}，{year}）\n"
        f"  · 切成 {len(chunks)} 个片段，写入集合 {PAPERS_COLLECTION}\n"
        f"  · 库容量 {before} → {after} 条（净增 {after - before}）\n"
        f"  · 现在可以用 query_corpus 检索这篇论文的正文了；"
        f"引用出处写 [{dest.stem}, p.页码]"
    )


# 所有工具的清单 —— 将来加工具，只改这一行
ALL_TOOLS = [query_corpus, search_arxiv, download_paper]
