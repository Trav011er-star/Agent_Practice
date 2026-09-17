import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

from torch import chunk

# 加载 .env 文件中的环境变量
load_dotenv(dotenv_path=".env")


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # 控制生成随机性
                temperature=temperature,
                # 流式输出
                # 生成一个迭代器，允许我们逐步获取模型的输出，而不是等待整个响应完成。
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            #             qwen2.5模型
            #                   |
            #                   | 生成token
            #                   ↓
            #             Ollama Server
            #                   |
            #                   | stream HTTP响应
            #                   ↓
            #             OpenAI Python SDK
            #                   |
            #                   | 自动封装成chunk对象
            #                   ↓
            #             你的for循环
            collected_content = []
            # response 不是一个已经装满数据的列表，而是一个连接着网络流的迭代器。
            # 每次遍历response时，都会从网络流中获取新的数据块（chunk），
            # 这些数据块包含了模型生成的部分输出。
            # 如果网络延迟没读到数据，for循环会等待 (response.next()) ，
            # 直到有新的数据块可用。

            # for chunk in response:
            # SDK 自动调用迭代器的next()方法，阻塞等待新的数据块。
            # 模型生成结束后，服务器通过网络发送一个“结束信号”，
            # OpenAI SDK 收到这个信号后，让迭代器进入结束状态，
            # 下一次 next() 抛出 StopIteration
            #     if not chunk.choices:
            #         continue
            #     # OpenAI格式
            #     #  chunk
            #     #   |
            #     #   └── choices
            #     #          |
            #     #          └── delta
            #     #                |
            #     #                └── content
            #     content = chunk.choices[0].delta.content or ""
            #     print(content, end="", flush=True)
            #     collected_content.append(content)
            # print()  # 在流式输出结束后换行
            # return "".join(collected_content)

            # 这样写就能直观感受到，大模型是一个个token生成并立刻流式传输回来的
            # 手动调用迭代器的next()方法，阻塞等待新的数据块。
            # 模型生成结束后，服务端发送流结束标记。
            # OpenAI SDK解析该标记，
            # 使迭代器耗尽，next()触发StopIteration异常。
            while True:
                try:
                    # 迭代器的 next() 方法会阻塞，直到有新的数据块可用。
                    chunk = next(response)
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    # flush=True 的作用是：
                    # 强制 Python 立即把输出显示到屏幕，而不是先暂存在输出缓冲区里。
                    print(content, end="", flush=True)
                # 模型生成结束后，服务器通过网络发送一个“结束信号”，
                # OpenAI SDK 收到这个信号后，让迭代器进入结束状态，
                # 下一次 next() 抛出 StopIteration
                except StopIteration:
                    print("\n大模型输出结束，迭代器已耗尽。")
                    break

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)


# >>>
# --- 调用LLM ---
# 🧠 正在调用 xxxxxx 模型...
# ✅ 大语言模型响应成功:
# 快速排序是一种非常高效的排序算法...
