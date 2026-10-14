"""
./venv/Scripts/python.exe -m uvicorn api:app --reload
把 Agent 从「命令行脚本」变成「HTTP 服务」。

────────────────────────────────────────────────────────────
它和 agent/graph.py 是什么关系？
────────────────────────────────────────────────────────────
        【命令行入口】                 【HTTP 入口】
        python agent/graph.py          POST /ask
                 ↓                          ↓
        graph.py 的 __main__ 块        api.py 的 query()
                 └────────────┬─────────────┘
                              ↓
                       graph.invoke(...)
                    （同一个图，同一套 Agent）

两个入口，喂给同一个图。这就是为什么初始 state 要抽成
make_initial_state() —— 免得两处各写一份，改一个忘另一个。
────────────────────────────────────────────────────────────

【怎么跑】
    ./venv/Scripts/python.exe -m uvicorn api:app --reload
                                    └─┬─┘ └─┬─┘
                                   文件名  里面那个变量名
    （翻译：在 api.py 里，找名叫 app 的那个对象）

然后浏览器打开 http://127.0.0.1:8000/docs —— FastAPI 白送的测试页。

────────────────────────────────────────────────────────────
【uvicorn 和 FastAPI 各管什么】

  uvicorn   服务器，管【网络】那层：把 HTTP 请求翻译成 Python 对象交给 app，
            再把 app 的返回翻译回 HTTP。
            ⚠️ 它只认识 app 这一个对象 —— 不知道有 /ask，也不知道有 query()。

  FastAPI   框架，管【业务】那层：拿着 app 里的路由表，
            决定哪个请求该进哪个函数。
────────────────────────────────────────────────────────────
"""

# ============================================================
# 第 1 件事：import
# ============================================================
from fastapi import FastAPI  # Web 框架本体
from pydantic import BaseModel  # "请求体长什么样"的声明工具
import sys
from pathlib import Path
from contextlib import asynccontextmanager  # 用来写 lifespan（第 2 件事讲）
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import traceback
import json

# ------------------------------------------------------------
# 手动把 agent/ 目录插到搜索路径最前面
# ------------------------------------------------------------
# 【为什么需要这行】
#   api.py 在 paper-Agent/ 下，而 graph.py 在 paper-Agent/agent/ 下 —— 差一层。
#   Python 导入时只在【搜索路径】里找，"agent" 这个目录默认不在里面，
#   所以 from graph import ... 会直接找不到。
#
# 【拆开读这行】
#   __file__                  → 本文件（api.py）
#   Path(__file__).resolve()  → 变成绝对路径
#   .parent                   → 上一级目录 → E:\Agent_Practice\paper-Agent
#   / "agent"                 → E:\Agent_Practice\paper-Agent\agent
#   sys.path.insert(0, …)     → 插到搜索路径【最前面】，优先找它
#
# 【为什么不写死 "E:\\Agent_Practice\\..."】
#   写死了别人 clone 下来路径就不一样，项目直接跑不起来。
#
# ⚠️ 这行必须在 from graph import ... 之前 —— 顺序反了就找不到。
sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))

# import graph 时就会把大模型先调用好
from graph import graph, make_initial_state  # noqa: E402
from engine import get_engine  # noqa: E402

#   graph              → graph.py 末尾 compile() 出来的那个【编译好的图】
#   make_initial_state → 图的初始输入怎么造（命令行和服务共用这一处）
#   get_engine         → 引擎（向量库 + BM25 + 重排模型）的构造函数
#
# 关于上面那条注释"import graph 时就会把大模型先调用好"：
#   对，graph.py 里 _llm_json / _llm_text 是【模块级】变量，
#   所以一 import 就建好了两个 LLM 客户端（日志里 [LLM] 会打印两遍）。
#   对服务来说这【正是想要的】：启动时建好，而不是每个请求建一遍。
#   附带好处是 fail fast —— .env 配错的话服务【压根起不来】，
#   而不是接了一堆请求才开始报错。
#
# 注：这两行会被编辑器标 E402（import 不在文件顶部）—— 那是必然的，
#     因为我们故意要先执行 sys.path.insert。用 noqa 让它闭嘴。


# ============================================================
# 第 2 件事：lifespan —— 服务的"开门前备菜"
# ============================================================
# 【要解决什么问题】
#   引擎（向量库 + 重排模型）加载要十几秒。
#   如果不提前加载，就会变成"第一个来问的人承担全部等待时间"，
#   而且日志会骗人：服务明明打印了 "Application startup complete"，
#   引擎却还没加载 —— 运维看到"启动完成"就往里导流量，导进来就卡住。
#
# 【lifespan 是什么】
#   一个钩子函数：服务【启动时】跑一次，【关闭时】也跑一次。
#   yield 是那条分界线：
#
#       上半段  →  服务启动时执行
#       ──── yield ────
#       下半段  →  服务关闭时执行
#
#   打个比方：饭馆开门前先【备菜】，打烊后【收拾】。
#             yield 就是"开门"那一刻。
#
# 【@asynccontextmanager 是干嘛的】
#   它是把"一个能 yield 的函数"包装成"上下文管理器"的装饰器。
#   FastAPI 规定 lifespan 必须长这样 —— 照抄格式即可，不用深究。
#   （async / await 是异步语法；这里只是格式要求，
#     里面调 get_engine() 这种同步函数完全没问题。）
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== 启动时（yield 之前）=====
    #   这里做"重活"：把引擎先建好。
    #   get_engine() 内部有缓存（`if _retriever is None`）：
    #     第一次调用 → 真的去加载
    #     之后再调用 → 直接返回，打印 "[engine] 已经就绪"
    #   所以放这儿重复调用是安全的。
    #
    #   顺带解决了一个隐患：如果不在启动时建好，
    #   启动瞬间来 3 个请求，3 个线程可能【同时】发现 _retriever is None，
    #   各造一遍引擎（engine.py 那个懒加载不是线程安全的）。
    #   先在这儿单线程建好，请求来了永远看到"已经有了"。
    print("服务要起来了，先把重活干了")
    get_engine()  # ← 就这一行：把引擎先建好

    yield  # ← 【分界线】上面是启动，下面是关闭

    # ===== 关闭时（yield 之后）=====
    #   本项目没有需要清理的资源，留个打印当"这条路是活的"的证据。
    #   （真要连数据库/线程池，就在这儿关掉。）
    print("服务要关了")


# ============================================================
# 第 3 件事：建服务对象，并把 lifespan 挂上去
# ============================================================
#   把这个 app 打包成一个 API，提供给 HTTP 调用。
#   app 就是"这个服务"本身，后面所有路由都登记到它身上。
#   （变量名 app 不能随便改 —— 启动命令 api:app 就是按这个名字找的。）
app = FastAPI(lifespan=lifespan)


# ============================================================
# 第 4 件事：声明"请求体长什么样"
# ============================================================
#   【怎么向服务提问】
#   POST /ask       ← 门牌号
#   请求体（JSON）   ← 内容
#       {"question": "Compare the pretraining objectives of BERT and GPT-3"}
#
#   这个类就是在说：请求体必须是个对象，里面有个 question 字段，类型是字符串。
#
#   【它白帮你干三件事】（一行校验代码都不用写）
#     ① /docs 里自动长出输入框 —— 不用写前端
#     ② 自动校验：少传 question、或传成数字 → 自动返回 422，
#        你的 ask() 函数【压根不会被调用】
#     ③ 函数里可以放心用 req.question —— 类型一定是 str
#
#   ⚠️ 422 和 500 是两回事：
#        422 = "你的请求有问题"（客户端错）
#        500 = "我的代码有问题"（服务端错）
#     让框架在门口拦下坏请求，就是这一步的价值。
class AskRequest(BaseModel):
    question: str


# ============================================================
# 第 5 件事：路由 —— 门牌号后面那个函数
# ============================================================
#   请求 "/ask"（POST 方法）时执行下面这个函数。
#
#   @app.post("/ask") 是【装饰器】，干的活是：
#     「把下面这个函数登记到 /ask 这扇门上，用 POST 方法进来」
#
#   get 和 post 的区别（HTTP 方法）：
#     GET  → 只读取数据，参数放在【网址】里（/hello?name=张三）
#     POST → 提交数据，数据放在【请求体】里
#   问题文本可能很长还带换行，塞网址里不合适，所以用 POST。
#
#   ⚠️ 注意是 def，不是 async def。
#      因为 graph.invoke() 是【同步】的（会卡住等 DeepSeek 回话，好几秒）。
#      写成 async def 会把整个服务卡死 —— 一个人提问，别人全得等。
#      写成普通 def，FastAPI 会自动把它扔到【线程池】里跑，互不阻塞。
@app.post("/ask")  # 对外，外部的 POST 请求 /ask 会执行下面的函数
def query(req: AskRequest):  # 对内，外部的 POST 请求 /ask 可以和函数名不同的
    # 参数类型写成 AskRequest → FastAPI 就知道"去请求体里找，按这个模型解析"
    # 参数名字随便取（req / body / data 都行），类型必须写对。

    question = req.question  # 取出来，纯粹为了下面读起来短一点

    # --------------------------------------------------------
    # 调 Agent。make_initial_state 造出图的初始输入。
    # --------------------------------------------------------
    #   graph.invoke(X) 里的 X 就是"初始 state" —— 图的【输入】。
    #
    #   为什么必须给这一整份，而不是只给 question？
    #   因为图里的节点一上来就要读这些字段：
    #       node_collect 读 hops / query / seen / items / tried / citations
    #   少给一个就是 KeyError。
    #
    #   （State 里一共 10 个字段，但有 2 个不用给：
    #       enough —— judge 会写它，写完了才有人读
    #       answer —— done 写了就结束了，没人读
    #     所以正好 8 个。这也是为什么 make_initial_state 里是 8 个键。）
    #
    #   ⚠️ 这两个数字是 10-13 加的 weak_rounds 改的。
    #      原来写的是 9 / 7 —— 加字段时忘了改注释。
    #      所以改 State 的时候记得回来看一眼这儿。
    result = graph.invoke(make_initial_state(question))

    # --------------------------------------------------------
    # 返回。return 字典 → FastAPI 自动转成 JSON 发回去。
    # --------------------------------------------------------
    #   为什么带上 hops（跳了几轮）？
    #     面试官看多跳 Agent，最大疑问是"你怎么证明它真的跳了"。
    #     answer 证明不了 —— 那只是一段文字。
    #     hops: 2 能证明 —— 它是运行结果的直接证据：
    #       一眼看出这道跨论文的题，Agent 自己迭代了 2 轮才凑齐证据。
    #
    #   （所以测的时候要用【跨论文】的题。单篇论文能答的题永远 hops: 1，
    #     展示不出任何东西 —— 见 worklog-2026-10-10 §13.5。）
    #
    #   graph.invoke 一旦抛异常，会一路冒到最外层，被下面那个
    #   @app.exception_handler(Exception) 接住 ——
    #   用户拿到体面的 500，服务端留下完整堆栈。见文件末尾「异常处理」。
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "hops": result["hops"],
    }


# 浅检查（liveness）：只回一句"我还活着"，不查任何东西。
#   为什么不写"深检查"（顺便 ping 一下引擎/大模型）？
#   因为上面 lifespan 已经预热了引擎 —— 服务能起来 = 引擎已就绪。
#   所以"能访问到 /health" 就等价于"能干活"。这是预热白送的好处。
#   另外：async def 是因为这里没有阻塞调用，不占线程池。
@app.get("/health")
async def liveness():
    return {"status": "ok"}


# --------------------------------------------------------
# 异常处理
# --------------------------------------------------------
# 注册一个 FASTAPI 的异常处理
@app.exception_handler(Exception)
# async：里面没有阻塞调用，不占线程池
async def handle_exception(request, exc):
    # ① 对内：留痕（服务端自己看）—— 三样都要
    #    谁进来的、什么病、怎么走到这儿的
    #    ⚠️ 异常【类型】是铁律，少了它，自己的笔误也会被静默吞掉
    print(f"[api] {request.method} {request.url.path}")  # → [api] POST /ask
    print(f"异常信息:{type(exc).__name__}: {exc}")
    traceback.print_exc()  # 完整堆栈 → 给日志/控制台
    # ② 对外：体面（用户看）
    #    只给一句人话，【不】暴露堆栈/文件路径/内部细节
    #    ⚠️ 返回值必须是 Response 对象，不能是 dict
    #       （Starlette 会拿返回值当 Response 调用 → 'dict' object is not callable）
    return JSONResponse(status_code=500, content={"error": "服务内部错误，请稍后重试"})


def _sse(payload: dict) -> str:
    """把一个 dict 打包成一帧 SSE 文本。

    纯搬运工，不含任何逻辑。抽出来唯一的原因是：
    下面要 yield 好几处，每处都手写 "data: " + json.dumps(...) + "\n\n"，
    迟早会漏那个换行 —— 而漏换行是最难查的一类错。
    """
    # 三块拼起来：
    #   ① "data: "  —— 固定开头。⚠️ 冒号后面【有一个空格】，别丢
    #   ② json.dumps(..., ensure_ascii=False)
    #        ensure_ascii 默认是 True，会把中文转成 \u4f60\u597d 这种转义。
    #        能用，但调试时看不懂。设 False 就保留原样中文。
    #   ③ "\n\n"    —— 两个换行 = 这条消息结束
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    def event_gen():
        hops = 0
        try:
            # 每走完一个节点 node 就停下来，把这一小步交给你，然后接着走
            # 每跑完一个节点，就执行一次循环
            # 图走到 END，就不再吐了。循环自动停
            # 图停着不动 ──► 你伸手要一次 ──► 它往前跑【一个节点】──► 停下 ──► 交给你
            for chunk in graph.stream(
                make_initial_state(req.question),
                # "updates" = 只给这个节点刚挪动的那部分
                # "values" = 每步把完整 state全给你
                stream_mode="updates",
            ):
                # chunk 是 {"节点名": {那一节点return的东西}}
                #   按名字分三种，用 _sse(...) 包好 yield 出去：
                #     "collect" → {"type":"progress","round":hops,"hits":len(citations)}
                #     "judge"   → {"type":"plan","next":query}
                #     "done"    → {"type":"final","answer":answer,"hops":hops}
                #   （hops 在循环外记着，因为 done 那帧自己不带轮数）
                for node, update in chunk.items():
                    # 每次返回的内容格式： {  节点名字  :  这个节点 return 的那个 dict  }
                    # chunk.items() 把它拆成 node 和 update
                    # node = "collect" / "judge" / "done"
                    # update = 这个节点 return 的那个 dict（节点 return 出来的 dict 本身）
                    if node == "collect":
                        hops = update["hops"]
                        yield _sse(
                            {
                                "type": "progress",
                                "round": hops,
                                "hits": len(update["citations"]),
                                "citations": update["citations"],
                            }
                        )
                    elif node == "judge":
                        if not update["enough"]:
                            yield _sse({"type": "plan", "next": update["query"]})
                    elif node == "done":
                        yield _sse(
                            {
                                "type": "final",
                                "answer": update["answer"],
                                "hops": hops,
                            }
                        )
        except Exception as e:
            print(f"[api] /ask/stream 中途出错：{type(e).__name__}: {e}")
            traceback.print_exc()
            yield _sse({"type": "error", "message": "服务内部错误，请稍后重试"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/")
def index():
    """把前端页面挂到根路径：浏览器打开 http://127.0.0.1:8000 就是它。"""
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")
