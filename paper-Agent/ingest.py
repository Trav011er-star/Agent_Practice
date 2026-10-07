"""建库脚本：PDF -> 分块 -> 向量化 -> 存入 Chroma

用法：
    # 用默认配置建库
    python ingest.py

    # 指定 chunk_size 建库（做对比实验时用）
    python ingest.py --chunk-size 500

    # 先清空同名集合再建，避免重复写入（解决原来"跑两次数据翻倍"的问题）
    python ingest.py --chunk-size 500 --rebuild

    # 一次建好三组，方便后面跑消融实验
    python ingest.py --chunk-size 500 1000 2000 --rebuild
"""

import argparse
import sys

import config
from src.embedding import count_chunks, create_vector_store, delete_collection
from src.health import OllamaNotReady, ensure_for
from src.loader import load_pdf
from src.splitter import split_documents


def ingest_one(chunk_size, rebuild=False):
    collection_name = config.collection_name_for(chunk_size)

    documents = load_pdf(str(config.DEFAULT_PDF))

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # 给每个片段一个稳定唯一的 chunk_id。
    # 混合检索要把"向量检索"和"BM25"两路结果对齐，必须有一个共同主键；
    # 只靠文本内容比对是不可靠的（不同段落可能有相同文字）。
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"c{chunk_size}_{i:04d}"

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
        nargs="+",
        default=[config.CHUNK_SIZE],
        help="要建库的 chunk_size，可以传多个（做消融实验）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="建库前先删除同名 collection，避免重复写入",
    )
    args = parser.parse_args()

    # 先自检，避免跑到一半才发现 Ollama 没启动
    try:
        ensure_for(need_llm=False)
    except OllamaNotReady as exc:
        print("\n[启动自检失败]\n", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"PDF: {config.DEFAULT_PDF.name}")
    print(f"向量库目录: {config.VECTORSTORE_DIR}")
    print(f"嵌入模型: {config.EMBEDDING_MODEL}")
    print()

    for chunk_size in args.chunk_size:
        ingest_one(chunk_size, rebuild=args.rebuild)

    print("\n建库完成。")


if __name__ == "__main__":
    main()
