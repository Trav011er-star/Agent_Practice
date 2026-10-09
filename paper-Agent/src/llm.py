from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import config

#   **kwargs = 「来者不拒」的口袋

#   def g(**kwargs):
#       print(kwargs)        # kwargs 是一个【字典】

#   g(a=1, b=2, c=3)
#   # → {'a': 1, 'b': 2, 'c': 3}

#   **kwargs 的作用：把所有"多出来的、我不认识的"关键字参数，统统收进一个字典。


# ┌────────────────────────────┬────────────────────────────┬──────┐
# │       ** 出现的位置         │           干什么           │ 叫法 │
# ├────────────────────────────┼────────────────────────────┼──────┤
# │ 定义函数时 def f(**kwargs)  │ 把散的关键字参数打包成字典   │ 收集 │
# ├────────────────────────────┼────────────────────────────┼──────┤
# │ 调用函数时 f(**d)           │ 把字典拆开成关键字参数       │ 拆包 │
# └────────────────────────────┴────────────────────────────┴──────┘

# f(d)      # 💥 把整个字典当成【一个】参数 → f(a={'a':1,'b':2}) → 缺 b，报错
# f(**d)    # ✅ 把字典【拆开】→ 等价于 f(a=1, b=2)


def get_llm_model(provider="ollama", **kwargs):
    # provider="deepseek" → 走 DeepSeek API
    # provider="ollama"   → 走本地（默认，保持向后兼容）**kwargs):
    # kwargs 原样透传给 ChatOllama（比如 format="json"）
    print("[LLM] 开始调用大模型 ...")
    print(f"[LLM] provider = {provider}")
    if provider == "deepseek":
        # 防御：没读到密钥就直接报清楚，别让它在网络请求里炸得莫名其妙
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("没读到 DEEPSEEK_API_KEY")

        # 把 Ollama 中对大模型返回格式设置形式转成 OpenAI 的设置形式
        format = kwargs.pop("format", None)
        if format:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        llm = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            **kwargs,
        )
        print(f"[LLM] 大模型就绪")
        return llm

    # 兜底走本地
    llm = ChatOllama(
        model=config.LLM_MODEL,
        # 显式指定地址，避免被环境变量 OLLAMA_HOST 带偏（详见 config.py 注释）
        base_url=config.OLLAMA_BASE_URL,
        **kwargs,
    )
    print(f"[LLM] 大模型就绪")
    return llm
