from src.loader import load_pdf
from src.splitter import split_documents
from src.embedding import create_vector_store, load_vector_store
from src.llm import get_llm_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

# 导入消息类型：
# HumanMessage 表示用户发送的消息；
# AIMessage 表示大语言模型回复的消息。
# 它们常用于保存和传递多轮对话历史。
from langchain_core.messages import HumanMessage, AIMessage

# 导入“消息占位符”：
# 在提示词模板中预留一个位置，用来动态插入聊天记录等消息列表。
from langchain_core.prompts import MessagesPlaceholder

# RAG_v2: Retrieval-Augmented Generation
# 多轮 RAG：第二轮时， llm 能看到第一轮记录

pdf_path = "data/papers/Attention Is All You Need.pdf"


def calculate_average(numbers):
    total = sum(numbers)


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


def RAG_workflow():

    # 保存当前会话的历史消息
    chat_history = []

    # 加载向量库和大模型
    vector_store = load_vector_store()
    llm = get_llm_model()

    # 检索最相关的 3 个 chunk
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Prompt 中插入历史对话：在工作流中，接收一个字典，并将键的对应值填入
    prompt = ChatPromptTemplate.from_messages(
        [
            # 1
            (
                "system",
                """
                    You are a paper reading assistant.

                    Please answer the question only based on the provided context.

                    If the context does not contain enough information,
                    say that the context is insufficient.

                    Context:
                    {context}
                """,
            ),
            # 2
            # ----------------------------------------------------
            # 历史聊天记录插入位置
            # ----------------------------------------------------
            #
            # 假设 chat_history 是：
            #
            # [
            #   HumanMessage("What is multi-head attention?"),
            #   AIMessage("Multi-head attention is ...")
            # ]
            #
            # 那么这里会把这两条消息作为真正的历史消息插入 Prompt。
            MessagesPlaceholder(variable_name="chat_history"),
            # 3
            (
                "human",
                "{question}",
            ),
        ]
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given the chat history and the latest user question,
                rewrite the latest question into a standalone question
                that can be understood without the chat history.

                Do not answer the question.
                Only return the rewritten question.
            """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "{question}",
            ),
        ]
    )

    # 从 chain 输入字典中取出字段
    get_question = RunnableLambda(lambda x: x["question"])

    get_history = RunnableLambda(lambda x: x["chat_history"])

    get_retrieval_query = RunnableLambda(lambda x: x["retrieval_query"])

    # 将检索到的多个 Document 拼成 context 字符串
    format_docs = RunnableLambda(
        lambda docs: "\n\n".join(doc.page_content for doc in docs)
    )

    # 改写问题 工作流，负责方便 retriever 理解历史提问
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # RAG 工作流
    chain = (
        {
            # 相同的输入（input 字典）这里有 3 条支路
            "context": get_retrieval_query | retriever | format_docs,
            "question": get_question,
            "chat_history": get_history,
        }
        # 整个字典作为上一部分的输出，输入给下一个组件
        | prompt
        | llm
        | StrOutputParser()
    )

    # 多轮对话
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        # 根据历史聊天记录修改当前的提示词，保证历史记录能被 retriever 了解
        # 例如把 it -> 具体的指代
        rewrite_input = rewrite_chain.invoke(
            {
                "question": query,
                "chat_history": chat_history,
            }
        )

        print("\n===== Query Rewrite =====")
        print("Original :", query)
        print("Rewritten:", rewrite_input)

        chain_input = {
            "question": query,  # 原始问题
            "retrieval_query": rewrite_input,  # llm 重写后给 Retriever 的问题
            "chat_history": chat_history,
        }

        response = chain.invoke(chain_input)

        print("\nAI:")
        print(response)

        # 把这一轮对话加入历史
        chat_history.append(HumanMessage(content=query))

        chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    RAG_workflow()
