from langchain_ollama import ChatOllama

import config


def get_llm_model():
    llm = ChatOllama(
        model=config.LLM_MODEL,
        # 显式指定地址，避免被环境变量 OLLAMA_HOST 带偏（详见 config.py 注释）
        base_url=config.OLLAMA_BASE_URL,
    )
    return llm
