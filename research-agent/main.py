"""最小的 Tool Calling 循环。

对照 05-ReAct智能体.py（手搓版）看差异：
    手搓版：模型吐一段文本 → 我拿正则解析出 Action → 我执行
    这 版：模型返回结构化字段 tool_calls → 我直接读 → 我执行
          （正则那一层被整个删掉了，不是改好了，是不需要了）

跑：python main.py
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

import settings
from tools import ALL_TOOLS

load_dotenv(settings.ENV_FILE)

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0,          # 判断类任务不要抖
)

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}      # 名字 → 函数，方便按名字找

# ⚠️ 绑定只做一次，但【每一轮请求都要带上它】。
#    见下面 TODO ① 的坑。
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def run(question: str, max_steps: int = settings.MAX_STEPS) -> str:
    """问一个问题，让模型自己决定调不调工具，最后返回它的答案。"""

    messages = [HumanMessage(question)]

    for step in range(1, max_steps + 1):
        # TODO ①：发出请求。
        #   提示：这里该用 llm 还是 llm_with_tools？
        #   想清楚为什么——tools 是【每次请求的入参】，不是"绑一次就记住"的设置。
        raise NotImplementedError("TODO ①")

        print(f"\n=== 第 {step} 轮 ===")
        print("模型说：", (ai.content or "").strip() or "（没说话，直接点单）")
        print("点了几单：", len(ai.tool_calls))

        # TODO ②：先把模型的回复记进 messages。
        #   不记会怎样？——下一轮模型不知道自己刚点过单，
        #   以为这是全新的一轮对话，于是原样再点一次。这就是"复读"，
        #   和 paper-agent §7.3 ② 那个 judge 复读是同一个病。
        raise NotImplementedError("TODO ②")

        # TODO ③：出口。
        #   模型"不点单了"代表什么？想清楚再写。
        #   （提示：它不点单 = 它认为手上的信息够回答了。
        #     所以这时候该把它的 content 返回出去，而不是继续循环。）
        raise NotImplementedError("TODO ③")

        # TODO ④：执行它点的单，把结果塞回 messages。
        #   - 用它给的工具名去 TOOLS_BY_NAME 里找函数
        #   - 用 .invoke(call["args"]) 执行（注意不是自己手动拆参数）
        #   - 包成 ToolMessage 追加进 messages
        #
        #   ⚠️⚠️ 最容易漏的一步：ToolMessage 必须带上 tool_call_id=call["id"]。
        #        漏了 OpenAI 系 API 直接报错。
        #        想一想为什么必须带 —— 模型一口气点了 3 单，回来 3 条结果，
        #        它凭什么知道哪条对应哪一单？（靠 id，不靠顺序）
        raise NotImplementedError("TODO ④")

    # 循环跑完 max_steps 还没出口 = 模型一直在点单
    return "（达到最大轮数，模型还在点单 —— 去看看是不是复读了）"


if __name__ == "__main__":
    question = "BERT 和 GPT-3 的预训练目标有什么区别？"
    print("问题：", question)

    answer = run(question)

    print("\n=== 最终答案 ===")
    print(answer)
