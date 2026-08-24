"""
pip install requests tavily-python openai
(1) requests 是从 Python 程序中访问网络 API 的 HTTP 库。
(2) tavily-python 是一个强大的 AI 搜索 API 客户端，用于获取实时的网络搜索结果，可以在官网注册后获取 API。
(3) openai是 OpenAI 官方提供的 Python SDK，用于调用 GPT 等大语言模型服务。
"""

# 提示词（用于交给 LLM 作为指令模板）
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

# 工具 1：查询真实天气
import requests


def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    它是一个开源的第三方天气查询服务，特点是：
    (1) 可以直接在浏览器或终端访问；
    (2) 支持通过城市名称查询天气；
    (3) 支持返回网页、文本、图片和 JSON
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()

        # 提取当前天气状况
        # 一些湿度、能见度之类的等级信息
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


import os
from tavily import TavilyClient


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    Tavily 可以根据提问抓取网页中的相关数据，整理成一个字典返回给客户端
    """
    # 1. 从环境变量中读取API密钥
    # 程序访问 Tavily 服务时使用的账号密码
    # api_key = os.environ.get("TAVILY_API_KEY")
    # if not api_key:
    #     return "错误:未配置TAVILY_API_KEY环境变量。"

    api_key = "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"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


if __name__ == "__main__":
    city = "东莞"
    weather = get_weather(city)
    print(weather)

    attraction = get_attraction(city=city, weather=weather)
    print(attraction)

    # 将所有工具函数放入一个字典，方便后续调用
    available_tools = {
        "get_weather": get_weather,
        "get_attraction": get_attraction,
    }
