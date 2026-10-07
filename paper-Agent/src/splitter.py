from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    把整篇论文切成一段段 chunk。

    chunk_size / chunk_overlap 不传就用 config 里的默认值；
    做对比实验时由 ingest.py 显式传进来。

    注意：split_documents 是"逐页切"的，
    所以每个 chunk 的 metadata['page'] 一定是单一页码，检索评测才能按页算命中。
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    return chunks
