from langchain_ollama import ChatOllama


def get_llm_model():
    llm = ChatOllama(
        model="qwen2.5:3b",
    )
    return llm
