import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
import settings
from tools import ALL_TOOLS

load_dotenv(settings.ENV_FILE)

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0,  # 判断类任务不要抖
)

SYSTEM_PROMPT = """你是一个论文问答助手，手上有一个本地论文库检索工具 query_corpus。

回答规则：
1. 涉及论文内容的问题，必须先用 query_corpus 检索原文，不许凭记忆作答。
   一个问题如果包含多个要点（比如要对比两篇论文），可以拆成几个针对性的子问题分别检索 ——
   这是允许的，也常常是必要的。
   但不要重复检索：如果某个内容上一次检索已经返回过，就不要换个说法再问一遍。
   与论文无关的问题（闲聊、写代码、时效性新闻）直接回答，不用调工具。
2. 答案里每一条来自检索的结论后面，都要标出处，格式 [论文名, p.页码]，
   且出处必须是工具返回里真实出现过的，不许自己编。
3. 如果 query_corpus 返回的片段不足以支撑回答，先试试 search_arxiv 去外部找；
    两边都找不到时才说「没有找到相关内容」，不要硬凑。
4. 允许补充你自己的通用知识，但必须明确标注「（通用知识，非论文原文）」。
"""

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}  # 名字 → 函数，方便按名字找

# ⚠️ 绑定只做一次，但【每一轮请求都要带上它】。
#    见下面 TODO ① 的坑。
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def run(question: str, max_steps: int = settings.MAX_STEPS) -> str:
    """问一个问题，让模型自己决定调不调工具，最后返回它的答案。"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for step in range(1, max_steps + 1):
        # TODO ①：发出请求。
        #   提示：这里该用 llm 还是 llm_with_tools？
        #   想清楚为什么——tools 是【每次请求的入参】，不是"绑一次就记住"的设置。
        # ai 的类型已经是 AIMessage
        ai = llm_with_tools.invoke(messages)

        print(f"\n=== 第 {step} 轮 ===")
        print("模型说：", (ai.content or "").strip() or "（没说话，直接点单）")
        print("点了几单：", len(ai.tool_calls))

        # TODO ②：先把模型的回复记进 messages。
        messages.append(ai)

        # TODO ③：出口。
        #   模型"不点单了" = 它认为手上的信息够回答了。
        #   这时候该把它的 content 返回出去，而不是继续循环
        if not ai.tool_calls:
            return ai.content
        # TODO ④：执行它点的单，把结果塞回 messages。
        #   - 用它给的工具名去 TOOLS_BY_NAME 里找函数
        #   - 用 .invoke(call["args"]) 执行（注意不是自己手动拆参数）
        #   - 包成 ToolMessage 追加进 messages
        #
        #   ⚠️⚠️ 最容易漏的一步：ToolMessage 必须带上 tool_call_id=call["id"]。
        #        漏了 OpenAI 系 API 直接报错。
        #        想一想为什么必须带 —— 模型一口气点了 3 单，回来 3 条结果，
        #        它凭什么知道哪条对应哪一单？（靠 id，不靠顺序）
        for call in ai.tool_calls:
            # ai.tool_calls 是个列表，因为模型可能一口气点多单。每一单长这样：
            # call = {
            #     "name": "query_corpus",
            # 传入工具函数的参数
            #     "args": {"question": "BERT 的预训练目标是什么"},
            #     "id":   "call_00_53xSxVozQ8Q0kB8mUlIl7341",
            #     "type": "tool_call",
            # }
            print("  → 调用", call["name"], "参数:", call["args"])
            # 工具是一个函数
            tool_fn = TOOLS_BY_NAME[call["name"]]
            result = tool_fn.invoke(call["args"])
            # id 是避免单词调用多次同一个工具，难以区分哪个回复是哪次的结果
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # 循环跑完 max_steps 还没出口 = 模型一直在点单
    return "（达到最大轮数，模型还在点单 —— 去看看是不是复读了）"


if __name__ == "__main__":
    question = "BERT 和 GPT-3 的预训练目标有什么区别？"
    print("问题：", question)

    answer = run(question)

    print("\n=== 最终答案 ===")
    print(answer)
