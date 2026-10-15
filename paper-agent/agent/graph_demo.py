"""
[工位1] ──传送带──> [工位2] ──传送带──> [工位3]

 - 工位 = 干一件事的地方 → 在代码里叫节点（node），本质就是一个函数
 - 传送带 = 决定下一步去哪 → 叫边（edge）
 - 传送带上流动的那个箱子 = 工位之间传递的数据 → 叫 state
"""

# ============================================================
# 第 1 件事：把要用的工具 import 进来
# ============================================================
# StateGraph : 用来"搭图"的模具
# START / END: 两个特殊标记，代表"起点"和"终点"
from langgraph.graph import StateGraph, START, END

# TypedDict : 一种写法，用来描述"这个盒子里有哪些字段"
from typing import TypedDict


# ============================================================
# 第 2 件事：定义 state（图里流动的那个箱子）
# ============================================================
# 这里先只放一个字段，名字叫 text，类型是字符串。
# 后面做真 Agent 时，这个箱子会变复杂（放问题、证据、答案……）。
# 读取数据仍然用字典方式 state["text"]
# class 子类(父类):
# 这里继承了字典类，并指定里面的 键
# 节点通过 state["键名"] 读取数据时，该键必须存在，否则会报 KeyError。
# 不要求传入字典的所有键与 State 完全一致。
class State(TypedDict):
    # 已经跳了几轮
    hops: int
    items: list[str]


MAX_HOPS = 5
# ============================================================
# 第 3 件事：节点
# 默认情况下，节点 return 中的同名字段会覆盖 State 中的旧值；没有返回的字段会保留
# 节点返回的字典，是覆盖，不是追加
# 假设箱子里现在是 items = ["旧条目"]


# return {"items": ["新条目"]}                    # ❌ 旧条目没了！变成 ["新条目"]
# return {"items": state["items"] + ["新条目"]}    # ✅ 旧的还在，变成 ["旧条目", "新条目"]
# ============================================================
# 节点 = 一个普通函数。规矩只有一条：
#   收一个 state 进来，返回一个 dict，表示"我要改哪些字段"。
# def node_a(state: State) -> dict:
#     print("① node_a 收到一条证据")
#     # print("node_a 拿到的 items =", state["items"])
#     # 返回的 dict 会被**合并**进 state。
#     # 这行的意思是：把 text 改成 "A 改过了"
#     return {
#         "text": "A 改过了",
#         "items": state["items"] + ["A 收集到的证据"],
#     }


# def node_b(state: State) -> dict:
#     print("② node_b 收到一条证据")
#     # 这里应该看到 A 改过之后的值 —— 这就是 state 的作用：
#     # 让后面的工位能看到前面工位留下的东西。
#     # print("node_b 拿到的 items =", state["items"])
#     return {
#         "text": "B 改过了",
#         "items": state["items"] + ["B 收集到的证据"],
#     }


def node_collect(state: State) -> dict:
    """收集一条证据。每被调用一次，就多一条。"""
    n = state["hops"] + 1  # 这是第几轮
    print(f"  [collect] 第 {n} 轮：收到一条证据")
    return {
        # 累加，注意这里加了"第几轮的证据"，方便看出是两个不同轮次加的
        "items": state["items"] + [f"第{n}轮的证据"],
        # hops 也要累加
        "hops": n,
    }


def node_judge(state: State) -> dict:
    """只报数，不改数据。"""
    print(f"  [judge] 现在有 {len(state['items'])} 条证据，已跳 {state['hops']} 轮")
    return {}


def node_done(state: State) -> dict:
    print("  [done] 够了，可以答了")
    return {}


# ============================================================
# 路由函数
# ============================================================
# 规矩：收一个 state，返回一个字符串（也就是"下一步去哪"）。
# 这个函数**不改数据**。
def route_after_judge(state: State) -> str:
    # # TODO: 写一个 if/else ——
    # #   如果证据数量 >= 2，返回 "enough"
    # #   否则返回 "not_enough"
    # if len(state["items"]) >= 2:
    #     return "enough"
    # return "not_enough"
    # # 返回的字符串必须和下面映射表左边的 key 一模一样（拼错会报错）

    # TODO：三种情况，按顺序判断（顺序很重要，想想为什么）
    #   1) 证据够了：len(state["items"]) >= 4      -> 返回 "go_done"
    #   2) 跳太多次了：state["hops"] >= MAX_HOPS  -> 返回 "go_done"   ← 刹车
    #   3) 以上都不是                              -> 返回 "go_again"
    #
    # 提示：三个 return，前面两个都带 if，最后一个是兜底。
    #       为什么"刹车"要放在"够不够"后面判断？（提示：够了的优先级更高）
    if len(state["items"]) >= 99 or state["hops"] >= MAX_HOPS:
        return "go_done"
    return "go_again"


# ============================================================
# 第 4 件事：把节点连成图
# ============================================================
builder = StateGraph(State)  # 建一个空图，并声明它的 state 用上面那个 State
# builder.add_node("a", node)  # 加一个节点，取名叫 "a"，内容是 node_a 这个函数
# builder.add_node("b", node_b)  # 再加一个，叫 "b"
# builder.add_node("judge", judge)
# builder.add_node("enough", node_enough)
# builder.add_node("not_enough", node_not_enough)

builder.add_node("collect", node_collect)
builder.add_node("done", node_done)
builder.add_node("judge", node_judge)

# builder.add_edge(START, "a")  # 起点 -> a
# builder.add_edge("a", "b")  # a -> b       ← 这就是"传送带"
# builder.add_edge("b", "judge")  # b -> 路由

builder.add_edge(START, "collect")
builder.add_edge("collect", "judge")

# 条件边
builder.add_conditional_edges(
    # ① 从哪个节点出来
    "judge",
    # ② 谁来决定下一步（路由函数）
    route_after_judge,
    # ③ 映射表：左边 = 路由函数的返回值，右边 = 真实节点名
    #    "这条边可能通向哪几个节点"。
    {
        "go_done": "done",
        "go_again": "collect",
    },
)
# 最后都接入到 END
# builder.add_edge("enough", END)
# builder.add_edge("not_enough", END)
builder.add_edge("done", END)

# compile() 可以理解成"检查图纸、准备开工"。
# 它会检查你连的边有没有指向不存在的节点之类的错误。
graph = builder.compile()

# ============================================================
# 第 5 件事：跑起来
# ============================================================
if __name__ == "__main__":
    print("开始运行")
    # invoke = 启动。参数是**初始 state**，也就是"箱子里一开始装什么"。
    # 传入的是一个字典
    # 输出的是 state
    result = graph.invoke(
        {
            "hops": 0,
            "items": [],
        }
    )
    print(result)
    print("运行结束")
