"""建库脚本：PDF -> 分块 -> 向量化 -> 存入 Chroma

两种语料，用 --corpus 选：

    single   只用 data/papers/Attention Is All You Need.pdf
             -> 建 attention_paper_<chunk_size>

    multi    data/papers 下所有 PDF（当前 5 篇，共 158 页）
             -> 建 papers_<chunk_size>

    两个库分工不同，都要留着：
      attention_paper_500  单论文，是评测基线（README 5.2/5.3 的数字来自它）
      papers_500           多论文，给 Agent 用（一篇论文撑不起多跳）

用法：
    # 建多论文库（Agent 用）—— 最常用
    python ingest.py --corpus multi --chunk-size 500 --rebuild

    # 重建单论文库（评测基线，一般不用动）
    python ingest.py --corpus single --chunk-size 500 --rebuild

    # 一次建好三组 chunk_size，方便跑消融实验
    python ingest.py --chunk-size 500 1000 2000 --rebuild

    # --rebuild 会先删除同名集合，避免重复写入
    # 不加则是【追加】，同一个库跑两次向量数会翻倍、检索结果出现大量重复
"""

import argparse
import sys
from pathlib import Path
import config
from src.embedding import count_chunks, create_vector_store, delete_collection
from src.health import OllamaNotReady, ensure_for
from src.loader import load_pdf
from src.splitter import split_documents


def collect_pdf_paths(corpus):
    """按 corpus 决定用哪些 PDF。
    single -> 只用 config.DEFAULT_PDF
    multi  -> data/papers 下所有 PDF（要排序，保证每次顺序一样）
    """
    # TODO 1: if/else，两行
    if corpus == "single":
        return [config.DEFAULT_PDF]
    return sorted(config.DATA_DIR.glob("*.pdf"))


def ingest_one(chunk_size, rebuild=False, corpus="multi"):
    collection_name = config.collection_name_for(chunk_size, corpus)

    # 把多篇 PDF 的 documents 的内容拼成一个 list
    documents = []
    for path in collect_pdf_paths(corpus):
        documents.extend(load_pdf(str(path)))

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # 给每个片段一个稳定唯一的 chunk_id。
    # 混合检索要把"向量检索"和"BM25"两路结果对齐，必须有一个共同主键；
    # 只靠文本内容比对是不可靠的（不同段落可能有相同文字）。
    for i, chunk in enumerate(chunks):
        paper_name = Path(
            chunk.metadata.get("source", "?")
        ).stem  # 去掉路径和 .pdf (只要主干 stem)
        chunk.metadata["chunk_id"] = f"c{chunk_size}_{i:04d}"
        chunk.metadata["paper"] = f"{paper_name}"

    if rebuild:
        delete_collection(collection_name)
    else:
        # 不加 --rebuild 就是在已有数据上追加。这是很容易踩的坑：
        # 同一个库跑两次，向量数量翻倍，而且检索结果会出现大量重复片段。
        # 这里主动提醒一下，而不是默默把库搞脏。
        try:
            existing = count_chunks(collection_name)
        except Exception:  # noqa: BLE001 —— 集合不存在时 chromadb 会抛各种异常，一律当 0 处理
            existing = 0
        if existing:
            print(
                f"  [警告] '{collection_name}' 里已有 {existing} 个片段，"
                f"这次是**追加**写入，不是覆盖。\n"
                f"         如果这不是你想要的，请加 --rebuild 重新建库。"
            )

    create_vector_store(
        chunks,
        collection_name=collection_name,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )

    print(
        f"  chunk_size={chunk_size:<5} 页数={len(documents):<3} "
        f"分块数={len(chunks):<4} -> collection '{collection_name}'"
    )
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="把 PDF 建成向量库")
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="+",  # nargs = "number of arguments"，意思是这个选项吃几个值。
        # ┌────────────┬────────────┬────────────────────────────────┐
        # │    写法     │   吃几个   │            结果类型            │
        # ├────────────┼────────────┼────────────────────────────────┤
        # │ 不写 nargs  │ 正好 1 个  │ 单个值 500                     │
        # ├────────────┼────────────┼────────────────────────────────┤
        # │ nargs="+"  │ 1 个或多个  │ list [500] / [500, 1000, 2000] │
        # ├────────────┼────────────┼────────────────────────────────┤
        # │ nargs="*"  │ 0 个或多个  │ list（可以是空的）             │
        # └────────────┴────────────┴────────────────────────────────┘
        default=[config.CHUNK_SIZE],
        help="要建库的 chunk_size，可以传多个（做消融实验）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="建库前先删除同名 collection，避免重复写入",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help='针对"single"单个或"multi"多个论文建库',
    )
    args = parser.parse_args()

    # 先自检，避免跑到一半才发现 Ollama 没启动
    try:
        ensure_for(need_llm=False)
    except OllamaNotReady as exc:
        print("\n[启动自检失败]\n", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)

    print("PDF:", [p.name for p in collect_pdf_paths(args.corpus)])
    print(f"向量库目录: {config.VECTORSTORE_DIR}")
    print(f"嵌入模型: {config.EMBEDDING_MODEL}")
    print()

    for chunk_size in args.chunk_size:
        ingest_one(chunk_size, rebuild=args.rebuild, corpus=args.corpus)

    print("\n建库完成。")


if __name__ == "__main__":
    main()
