from langchain_community.document_loaders import PyMuPDFLoader
from pymupdf import Document


def load_pdf(path):

    loader = PyMuPDFLoader(path)

    documents = loader.load()

    return documents


# documents: 一个List[Document]，
# 每个Document对象对应PDF中的一页
# |
# |-- Document 0  (第1页)
# |
# |-- Document 1  (第2页)
# |
# |-- Document 2  (第3页)
# |
# ...
# |
# |-- Document 14 (第15页)

# Document 对象
# Document(
# 【正文内容】
#     page_content="Attention is all you need...",
# 【附加信息（文件来源、页码、格式、关键词...）】
#     metadata={
#         "source": "Attention.pdf",
#         "page": 0
#     }
# )


def main():
    pdf_path = "data/papers/Attention Is All You Need.pdf"
    documents = load_pdf(pdf_path)

    print("文档页数:", len(documents))

    print("\n第一页内容:")
    print(documents[0].page_content[:500])

    print("\n元数据:")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()
