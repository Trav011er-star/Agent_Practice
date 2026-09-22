from src.loader import load_pdf
from src.splitter import split_documents
from src.embedding import create_vector_store, load_vector_store
from src.llm import get_llm_model
from langchain_core.prompts import ChatPromptTemplate

# 将 llm 的输出直接转为可读的字符串(否则需要 response.content)
from langchain_core.output_parsers import StrOutputParser

# Runnable：LangChain 对“可执行组件”的统一抽象。
# 一个 Runnable 可以接收输入、执行处理，并返回输出。

# LangChain 工作流中的各个组件通常都实现了 Runnable 接口，
# 因此可以统一使用类似 invoke() 的方法执行。

# 当执行 chain.invoke(...) 时，
# LangChain 会按照工作流顺序依次执行其中的各个 Runnable 组件。

# RunnableLambda：把普通 Python 函数包装成 Runnable，
# 使普通函数也可以接入 LangChain 的 LCEL 工作流。

# RunnablePassthrough：不对输入做任何处理，原样返回输入。
# 常用于工作流分支中，保留原始输入继续向后传递。
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

# RAG_v1: Retrieval-Augmented Generation
# 单轮 RAG

pdf_path = "data/papers/Attention Is All You Need.pdf"


# 1. 加载数据
def data_load(pdf_path=pdf_path):
    documents = load_pdf(pdf_path)

    print("文档页数:", len(documents))

    print("\n第一页内容:")
    print(documents[0].page_content[:500])

    print("\n元数据:")
    print(documents[0].metadata)

    return documents


# 2. 分割数据
def split_data(documents):
    chunks = split_documents(documents)

    # 52
    print("\n分块数量:", len(chunks))
    print("\n第一块内容:")
    print(chunks[0].page_content[:500])

    return chunks


# 3. 创建向量存储
def build_vectorStore(chunks):
    vector_store = create_vector_store(chunks)
    print("\n向量存储创建完成，已保存到 Chroma 数据库。")

    # 52
    print("\n向量存储中的向量数量:", len(vector_store.get()["ids"]))
    print("\n向量存储中的第一条向量:")
    # dict_keys(['ids', 'embeddings', 'documents', 'uris',
    # 'included', 'data', 'metadatas'])
    print(vector_store.get().keys())
    print(
        vector_store.get(
            # 指定读取 5 条向量
            # 指定读取哪些字段，这里只读取 embeddings 字段
            # 否则太大的话读取只能看到 None
            limit=5,
            include=["embeddings"],
        )["embeddings"][0]
    )

    return vector_store


def RAG_retrieval(query, vector_store):

    # RAG:根据 query 在向量数据库中检索前 k 个最相似的文档
    # 定义一个retriever，适用于多个向量数据库
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        print(f"\n===== Result {i + 1} =====")
        print(doc.page_content[:500])
        print(f"\npage = {doc.metadata['page']}, \nsource = {doc.metadata['source']}")

    # 将前 k 个检索相关的内容拼接起来
    context = "\n\n".join(doc.page_content for doc in results)

    return context


# RAG 分步骤流程
def RAG_step_by_step():
    # ----------------------------- 离线建库 -----------------------------
    # # 1. 加载数据
    # documents = data_load()

    # # 2. 分割数据
    # chunks = split_data(documents)

    # # 3. 创建向量存储
    # vector_store = build_vectorStore(chunks)
    # # 创建过一次后不要重复创建，每次写入向量存储不是重写而是添加

    # ----------------------------- 在线查询 -----------------------------
    # 查询（自然语言）
    query = "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. What is the meaning of this sentence?"

    # 加载向量存储
    vector_store = load_vector_store()

    # 4. RAG 检索:将 query 作为输入，检索出最相似的文档，
    context = RAG_retrieval(query, vector_store)

    # 5. 按照提示词模板和查询语句拼接成上下文
    prompt = ChatPromptTemplate.from_template("""
        You are a paper reading assistant.

        Please answer the question only based on the provided context.

        If the context does not contain enough information,
        say that the context is insufficient.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

    # message = prompt.invoke(
    #     {
    #         "context": context,
    #         "question": query,
    #     }
    # )

    # print(f"\n发送给llm：{message}")

    # 5. 调用大模型
    llm = get_llm_model()
    # chain：‘|’ 把前一个组件的输出，自动传给后一个组件
    chain = prompt | llm
    # 这步相当于 prompt.invoke + llm.invoke(message)
    # 将 RAG 检索的上下文内容和查询的自然语言装入提示模板中，
    # 并传递给 llm (体现Langchain 流水线)
    response = chain.invoke({"context": context, "question": query})
    # response = llm.invoke(message)
    print("\n===== LLM Response =====")
    print(response.content)


def RAG_workflow():
    # ----------------------------- 在线查询 -----------------------------
    # 查询（自然语言）
    query = "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. What is the meaning of this sentence?"
    # 加载向量存储
    vector_store = load_vector_store()
    # 加载大模型
    llm = get_llm_model()
    # RAG 检索器，返回与查询相关的前 k 个内容
    # Retriever 实现了 LangChain 的 Runnable 接口，
    # 所以在工作流中可以直接自动调用 .invoke()
    # invoke(query) 中会自动利用向量存储的 embedding 模型，
    # 将 query 转为向量
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # 提示词模板
    prompt = ChatPromptTemplate.from_template("""
        You are a paper reading assistant.
    
        Please answer the question only based on the provided context.
    
        If the context does not contain enough information,
        say that the context is insufficient.
    
        Context:
        {context}
    
        Question:
        {question}
    
        Answer:
        """)

    # Langchain 流水线
    # 前一个的输出作为后一个的输入
    chain = (
        # chain.invoke() 读取到的输入作为 retriever.invoke() 的输入
        {
            # Retriever 对象实现了 Runnable 接口，
            # 因此可以通过 invoke() 执行。
            # 当 Retriever 被放入 LangChain 工作流中时，
            # 执行 chain.invoke(...) 会自动触发 Retriever 的 invoke()。
            "context": retriever
            # 将 retriever.invoke() 的输出交给 RunnableLambda 包装的方法
            | RunnableLambda(
                lambda docs: "\n\n".join(doc.page_content for doc in docs)
            ),
            # RunnablePassthrough: 将 chain 的输入原样输入到这
            "question": RunnablePassthrough(),
        }
        # 将整个字典作为 prompt.invoke() 的输入
        | prompt
        # 将 prompt.invoke(dict) 的输出作为 llm.invoke() 的输入
        | llm
        # llm.invoke() 的输出作为 chain 的输出
        | StrOutputParser()
    )

    # 封装完成的 Langchain，只需给入 query 即可得到 RAG 的输出结果
    response = chain.invoke(query)
    print("\n===== LLM Response =====")
    print(response)


if __name__ == "__main__":
    # RAG_step_by_step()
    RAG_workflow()
