"""给片段分配稳定唯一的 chunk_id。

为什么单独成一个文件
--------------------
     ingest.py       全量建库，一次处理 data/papers 下所有 PDF
     download_paper  增量入库，一次只加一篇新下载的

两条路径都必须算出【同一套编号】。各写一份实现 = 迟早分叉 =
又回到"编号撞车、静默覆盖"。所以编号逻辑只能有一份，就是这里。
（和"工具的 docstring 只能有一份"是同一个道理。）

编号方案
--------
     c{chunk_size}_{论文 slug}_{论文内序号:04d}
     例：c500_bert-2018_0042

为什么序号要【按论文】计，不能全局连号
--------------------------------------
老写法是 f"c{chunk_size}_{i:04d}"，i 来自 enumerate(所有论文 chunks 拼成的大列表)。
也就是 Attention 占 0000~0128、BERT 接着 0129 起 —— 编号【取决于别的论文在不在】。
于是：

  · 新论文单独入库，编号从 0000 重新开始 → 静默覆盖掉 Attention
    （Chroma 对重复 id 是【覆盖】不是追加，实测：条数不变、内容被换掉、不报错）
  · 加一篇文件名排在字母序靠前的论文 → 它后面每篇的编号整体后移

改成带论文 slug 之后，编号只由论文自己决定，别人加不加都影响不到它。
副作用是"同一篇论文重复入库 = id 完全相同 = 覆盖"，这正是我们想要的幂等。
"""

import re
from pathlib import Path

# 把"非字母数字"的连续串压成一个 "-"，用来把论文名清成能塞进 id 的形式。
_SAFE_RE = re.compile(r"[^a-z0-9]+")


def slugify(name):
    """论文名 -> 能塞进 id 的短标识。

        "BERT (2018)"                -> "bert-2018"
        "Attention Is All You Need"  -> "attention-is-all-you-need"

    全中文文件名清完会剩空串，这时退回用原名。
    兜底必须是【确定且唯一】的：
      · 确定 —— 同一篇论文每次算出来都一样，否则重复下载就不是覆盖而是新增
      · 唯一 —— 写死一个 "untitled" 会让两篇中文论文算出同一个 slug，
                而 slug 是主键的一部分，撞了就等于串库
    直接用原名就同时满足这两条，而且不需要引入哈希（兜底场景不需要加密强度，
    Chroma 的 id 本来就是个任意字符串）。
    """
    slug = _SAFE_RE.sub("-", name.lower()).strip("-")
    return slug or name


def assign_chunk_ids(chunks, chunk_size):
    """就地为每个 chunk 补上 chunk_id 和 paper，返回同一个 list。

    调用方（ingest.py / download_paper）要拿 metadata["chunk_id"]
    作为写进 Chroma 的 id —— 混合检索靠它对齐"向量"和"BM25"两路结果。
    """
    counters = {}  # slug -> 这篇论文已经编到第几号

    for chunk in chunks:
        paper_name = Path(chunk.metadata.get("source", "?")).stem

        # 计数和 id 必须用【同一个键】。id 里放的是 slug，所以 counters 也按 slug 计，
        # 否则两边会错位。
        slug = slugify(paper_name)
        i = counters.get(slug, 0)
        counters[slug] = i + 1

        # 序号按【论文】计 —— 新论文永远从 0000 起，跟别的论文存不存在无关
        chunk.metadata["chunk_id"] = f"c{chunk_size}_{slug}_{i:04d}"

        # 注意：paper 这个字段【不】slug 化。
        # 它是给最终答案里的引用 [BERT (2018), p.3] 用的，给人看；
        # slug 是给机器当主键的。同一个名字，两种用途，两份形态。
        chunk.metadata["paper"] = paper_name

    return chunks
