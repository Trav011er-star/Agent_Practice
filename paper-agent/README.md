# Paper-Agent

**一个带多跳检索 Agent 的论文问答助手，打开浏览器就能试。** 把 PDF 论文建成向量库，
混合检索召回证据，再由 **LangGraph 多跳 Agent** 判断"证据够不够"，不够就自己改写
检索词再搜一轮，最后生成**每句都带出处**的答案。

已封装为 **FastAPI 服务**：`POST /ask/stream` 按节点粒度推送进度、答案逐字吐出，
根路径直接返回一个单页前端。检索层有 57 条标注问题的**可复现量化评测**（Recall@k / MRR），
并接入 **Langfuse** 观测每次提问的完整轨迹。

---

## 目录

| 章节 | 里面有什么 |
|:--|:--|
| [**1. 项目简介**](#1-项目简介) | [解决什么问题](#解决什么问题) · [一句话亮点](#一句话亮点) · [语料](#语料) |
| [**2. 核心功能**](#2-核心功能) | 检索层 / Agent 层 / 服务层分别实现了什么 |
| [**3. 技术栈**](#3-技术栈) | LangChain · Chroma · BM25 · LangGraph · DeepSeek · FastAPI · Langfuse |
| [**4. 系统架构**](#4-系统架构) | [建库（离线）](#41-建库离线跑一次) · [多跳 Agent（在线）](#42-多跳-agent在线) · [服务化架构](#43-服务化架构) · [目录结构](#44-目录结构) |
| [**5. 效果数据**](#5-效果数据) | [指标定义](#51-指标定义) · [chunk_size 消融](#52-chunk_size-消融实验) · [检索策略消融](#53-检索策略消融实验) · [多跳 Agent 的实际运行](#54-多跳-agent-的实际运行) · [性能：时间花在哪](#55-性能时间花在哪) |
| [**6. 复现流程**](#6-复现流程) | [环境准备](#61-环境准备) · [配置密钥](#62-配置密钥) · [建库](#63-建库) · [运行](#64-运行) · [评测](#65-评测) |
| [**7. 知识点、踩坑与注意事项**](#7-知识点踩坑与注意事项) | [7.1 一类高频 bug](#71--一类高频-bug加了个东西但没接到该接的地方) · [检索层](#72-检索层) · [Agent 层](#73--agent-层) · [环境层](#74--环境层) · [生成层 / 配置层](#75-生成层--配置层) · [服务层与可观测性](#76--服务层与可观测性) |
| [**8. 工作记录**](#8-工作记录) | 逐日进展与踩坑（`docs/worklog-*.md`） |

---

## 1. 项目简介

### 解决什么问题

直接问大模型"BERT 和 GPT-3 的预训练目标有什么区别"，它会一本正经地编。
把论文喂进去也一样有问题——**单轮 RAG 只搜一次**，遇到需要综合多篇论文、
或多个段落才能回答的问题，一次检索根本凑不齐证据。

这个项目做两件事：

1. **把检索做扎实**：不是"向量检索一把梭"，而是 BM25 + 向量混合召回、RRF 融合、
   cross-encoder 精排，每一层都用评测数据验证过（见 §5）。
2. **把"搜几次"交给模型决定**：用 LangGraph 搭一个多跳 Agent —— 搜一轮 →
   让大模型判断证据够不够 → 不够就生成新的检索词再搜 → 够了才作答。
   回答里每个事实都标注 `[论文名, p.页码]`，可以回溯到具体片段。

### 一句话亮点

> **多跳是否触发，取决于问题是否需要跨文档，而不是代码写得对不对。**
> 语料只有一篇论文时，多跳代码 100% 正确也永远只跑一轮。
> 这条结论是实际踩出来的，也是判断一个多跳 Agent 是否真在工作的方法（详见 §7.3）。

### 语料

5 篇 LLM 奠基论文，共 158 页 / 1687 个片段：

```text
Attention Is All You Need (2017) → BERT (2018) → GPT-3 (2020) → RAG (2020) → ReAct (2022)
```

---

## 2. 核心功能

| 能力 | 说明 | 位置 |
|---|---|---|
| PDF 建库 | PyMuPDF 按页解析 → 分块 → 向量化 → Chroma 持久化 | `ingest.py` |
| **混合检索** | 自实现 BM25 + 向量检索，用 **RRF** 融合（不是分数加权） | `src/retrieval.py` |
| **重排** | cross-encoder 精排，模型按语言/领域**实测选型** | `src/retrieval.py` |
| 单轮 RAG | LCEL 管道 `prompt \| llm \| StrOutputParser` | `RAG_v1.py` |
| 多轮 RAG | query rewrite（指代消解）+ 对话历史注入 | `RAG_v2.py` |
| **多跳 Agent** | LangGraph：检索 → 判断证据 → 改写 query → 再检索，`MAX_HOPS` 刹车 | `agent/graph.py` |
| **防复读** | `tried` 列表记录已问过的检索词并喂给 judge | `agent/graph.py` |
| **答案溯源** | 回答内嵌 `[论文名, p.页码]`，可回溯到片段级 `chunk_id` | `agent/graph.py` |
| **拒答** | 资料不足时明确说"资料里没有提到"，而不是编造 | `agent/graph.py` |
| **可复现** | judge/done 节点 `temperature=0`，同一问题多次运行结果一致 | `agent/graph.py` |
| 检索引擎缓存 | 多跳每一轮不重建 BM25 索引 / 重排模型 | `agent/engine.py` |
| **检索评测** | 57 条标注问题，Recall@k / AnswerHit@k / MRR | `eval/` |
| 模型可切换 | 本地 Ollama ↔ 云端 DeepSeek，改一行开关 | `src/llm.py` |
| **HTTP 服务** | FastAPI 封装：`POST /ask` 同步接口 + `POST /ask/stream` 流式接口 + `GET /health` | `api.py` |
| **SSE 流式** | 按 LangGraph 的**节点粒度**推送进度，前端可见每一轮推进与检索词改写 | `api.py` |
| **逐字输出** | `stream_mode=["updates","messages"]`，答案随生成一个字一个字吐出 | `api.py` |
| **单页前端** | `GET /` 直接返回页面，输入问题即看进度流 + 渲染后的答案与引用 | `static/index.html` |
| **服务预热** | `lifespan` 启动时先建好检索引擎，第一个请求不承担全部加载耗时 | `api.py` |
| **可观测性** | 每次提问的完整轨迹上报 Langfuse：每个节点耗时、LLM 调用、多跳结构 | `api.py` |

---

## 3. 技术栈

| 层 | 用了什么 | 为什么 |
|---|---|---|
| 编排 | **LangGraph** | 多跳循环是**有状态、带条件跳转**的流程，用图来表达比写 `while` 循环清晰得多 |
| 检索 | 自实现 **BM25** + 向量检索 + **RRF** 融合 | 专有名词/数字（`BLEU`、`WMT 2014`、`16,384`）向量模型不敏感，关键词匹配一抓一个准 |
| 重排 | **sentence-transformers** cross-encoder | 从 20 条候选里精挑 3 条 |
| 向量模型 | Ollama **`nomic-embed-text`**（本地，768 维） | 免 API、可离线 |
| 生成模型 | **DeepSeek**（云端）/ Ollama **`qwen2.5:3b`**（本地） | 可切换；实测判断力差距很大，见 §7.5 ③ |
| 向量库 | **Chroma**（本地持久化） | 轻量、免部署 |
| 解析/分块 | **PyMuPDF** + langchain `RecursiveCharacterTextSplitter` | 按页解析，保证 chunk 的页码唯一 |
| 评测 | 自实现 Recall@k / AnswerHit@k / MRR | 指标定义见 §5.1 |
| 服务 | **FastAPI** + **uvicorn** | 路由写成同步 `def`，uvicorn 自动扔进线程池，多个请求并行处理，不会一个人提问卡住所有人 |
| 流式协议 | 原生 **SSE**（`StreamingResponse`） | 比 WebSocket 简单得多，且天然单向——这里只需要服务端推、客户端不推 |
| 可观测性 | **Langfuse**（云版） | 通过 LangChain 的 `CallbackHandler` 挂进图，**不侵入业务代码** |
| 依赖管理 | `venv` + `requirements.txt` | |

---

## 4. 系统架构

### 4.1 建库（离线，跑一次）

```text
PDF ──PyMuPDFLoader──> 每页一个 Document
    ──RecursiveCharacterTextSplitter（逐页切，不跨页）──> chunks
    ──打标签（chunk_id / paper）──> chunks
    ──nomic-embed-text──> 768 维向量
    ──> Chroma（向量 + 原文 + metadata 三件一起存）
```

> **为什么逐页切？** 因为评测要按页算命中，跨页就不知道算哪一页。
> **为什么 `chunk_id` 要自己给？** 混合检索要把"向量路"和"BM25 路"的结果对齐，
> 必须有个共同主键——靠文本内容比对不可靠（不同段落可能有相同文字）。

### 4.2 多跳 Agent（在线）

```text
        ┌──────────────────────────────────────┐
        ↓                                      │ 不够：改写 query
START → collect（检索 + 去重） → judge（判断够不够）
                                     │
                                     └─ 够了 ─→ done（生成带出处的答案） → END
```

- **`collect`**：`retrieve()` 取片段 → 按 `chunk_id` 过滤掉 `seen` 里已有的 → 累加进 `items`
- **`judge`**：把「原问题 + 已收证据 + 已问过的检索词」给大模型，要它回 JSON
  `{"enough": bool, "next_query": str}`
- **`done`**：把证据给大模型，要求只依据资料作答、标注 `[论文名, p.页码]`

**三条出口**（不只有"够了"一条）：

| 出口 | 触发条件 | 意义 |
|---|---|---|
| 够了 | judge 判 `enough = true` | 主出口——包括"资料里确实没有"这种诚实答案 |
| 白跑了 | 连续 `MAX_WEAK_ROUNDS` 轮没捞到新的有效证据 | 保险丝：问下去也是白问 |
| 到顶了 | `hops >= MAX_HOPS` | 兜底，防止意外死循环 |

后两条是**搜索质量不好时提前止损**用的，理由和取值依据见 §7.3。

### 4.3 服务化架构

```text
浏览器                      FastAPI 服务                        LangGraph 图
  │                             │                                  │
  │  POST /ask/stream           │                                  │
  │  {"question": "..."} ──────►│  ask_stream() ──────────────────►│  graph.stream(
  │                             │                                  │    stream_mode=
  │                             │                                  │    ["updates",
  │                             │                                  │     "messages"])
  │                             │                                  │
  │                             │ ◄── updates：节点跑完 ───────────┤ → progress / plan / final
  │                             │ ◄── messages：吐一个 token ──────┤ → token
  │  ◄── SSE 逐帧推回 ──────────┘                                  │
  │      （每一帧 = `data: {...}` + 一个空行）                      │
  └─ 每收到一帧就更新页面
```

三层各管一段，互不知道对方内部：

| 层 | 管什么 | 关键点 |
|---|---|---|
| `uvicorn` | 网络层：HTTP ↔ Python 对象 | 只认识 `app` 这一个对象，不知道有 `/ask` |
| `FastAPI` | 路由层：哪个请求进哪个函数 | `AskRequest` 在**门口**校验请求体，格式坏了直接 422，业务函数**根本不执行** |
| `graph.stream` | 业务层：跑图并逐帧交出 | **拉模型**——图不会自己往前跑，是循环每要一次它才走一个节点 |

**一次提问会同时产生两种粒度的流**：节点级（几帧）和 token 级（几百帧）。
它们**汇进同一条管道**，每帧带一个 `mode` 标签说明自己属于哪一类。细节见 §7.6 ④。

两个设计选择值得单独说：

- **为什么用 SSE 而不是 WebSocket？** 这里只需要服务端单向下推，WebSocket 的双向能力和
  握手成本是白搭。SSE 就是一个普通 POST + `text/event-stream` 响应。
- **为什么路由写 `def` 而不是 `async def`？** `graph.invoke()` 是**同步阻塞**的
  （要等大模型好几秒）。写成 `async def` 会把整个事件循环卡死——一个人提问，所有人排队。
  写成普通 `def`，FastAPI 自动扔进线程池，互不阻塞。

### 4.4 目录结构

```text
paper-agent/
├── config.py              # 所有可调参数集中在此（含 LLM_PROVIDER 开关）
├── ingest.py              # 建库：PDF -> 分块 -> 向量化 -> Chroma
├── RAG_v1.py / RAG_v2.py  # 单轮 / 多轮 RAG
├── api.py                 # HTTP 服务入口：/ask、/ask/stream（SSE）、/health、/
├── requirements.txt
├── .env                   # 密钥（已 gitignore，需自行创建，见 6.2）
├── test_api.py            # 服务接口的回归测试脚本（curl 风格，非 pytest）
├── test_weak_rounds.py    # 检索降级路径的回归测试
├── agent/
│   ├── graph.py           # LangGraph 多跳 Agent：collect / judge / done
│   ├── engine.py          # 检索引擎的进程级缓存
│   ├── graph_demo.py      # 早期"假证据"版骨架（学习过程存档，非有效代码）
│   └── test.py            # 早期脚本（学习过程存档）
├── src/
│   ├── loader.py          # PDF 解析
│   ├── splitter.py        # 分块
│   ├── embedding.py       # 向量化 + Chroma 读写（含分批写入）
│   ├── llm.py             # 大模型工厂（provider 切换）
│   ├── retrieval.py       # 混合检索与重排
│   └── health.py          # 启动自检：Ollama 是否可用、模型是否已下载
├── static/
│   └── index.html         # 单页前端（fetch + ReadableStream 解 SSE；迷你 Markdown 渲染）
├── eval/
│   ├── gold_qa.json       # 评测集：57 条问题 + 标准答案页码 + 原文证据
│   ├── metrics.py         # Recall@k / AnswerHit@k / MRR
│   ├── run_eval.py        # 一键跑评测
│   ├── rerank_compare.py  # 对比不同重排模型
│   └── results/           # 评测结果存档
├── docs/                  # 逐日工作记录
├── data/papers/           # 论文 PDF（5 篇）
└── vectorstore/           # Chroma 数据库（已 gitignore，可重建）
```

---

## 5. 效果数据

### 5.1 指标定义

- **Recall@k**：前 k 个片段中，至少有一个来自正确页面的题目占比。衡量"该找的有没有捞出来"。
- **AnswerHit@k**：前 k 个片段中**真的包含标准答案原文**的题目占比（子串匹配）。
  比 Recall@k 严格得多——落在正确页码 ≠ 捞到了那句话。
- **MRR@10**：第一个命中片段的排名倒数平均值。越接近 1 说明正确答案排得越靠前。

### 5.2 chunk_size 消融实验

57 条问题（覆盖事实、数字表格、名词定义、公式四类），k=10，模型 `nomic-embed-text`，
纯检索评测不含生成。

| chunk_size | 片段数 | Recall@1 | Recall@3 | Recall@5 | AnswerHit@1 | AnswerHit@3 | AnswerHit@5 | MRR@10 |
|-----------:|-------:|---------:|---------:|---------:|------------:|------------:|------------:|-------:|
| **500** | 129 | **75.4%** | **91.2%** | **94.7%** | 50.9% | 73.7% | 77.2% | **0.841** |
| 1000 | 52 | 57.9% | 80.7% | 91.2% | 42.1% | 64.9% | 73.7% | 0.706 |
| 2000 | 27 | 54.4% | 84.2% | 93.0% | 45.6% | **75.4%** | **82.5%** | 0.702 |

> **可复现性的边界（实测）**：同一 Ollama 进程内连跑两次结果逐位一致；
> 但**换一个 Ollama 进程**后 **Recall@1 会抖约 ±5 个百分点**，
> 而 Recall@3 / Recall@5 / MRR 完全不动。
> 逐题对比发现变化的题目**全部**是"第 1 名 ↔ 第 2 名互换"——
> 检索到的内容没变，抖的只是近似并列时谁排前面。
> 疑似根因是不同进程的 embedding 浮点差异，**该假设未经证实**。
> 所以主结论请看 **Recall@3 / MRR**，Recall@1 只作参考。

### 5.3 检索策略消融实验

固定 `chunk_size=500`，混合检索 = 向量 + BM25 各召回 20 条，RRF（k=60）融合。

| 检索策略 | Recall@1 | Recall@3 | Recall@5 | AnswerHit@1 | AnswerHit@3 | AnswerHit@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense（纯向量） | 75.4% | 91.2% | 94.7% | 50.9% | 73.7% | 77.2% | 0.841 |
| hybrid（BM25 + 向量 + RRF） | 87.7% | **98.2%** | 98.2% | 61.4% | 86.0% | 89.5% | 0.924 |
| + 重排 `bge-reranker-base` | 78.9% | 93.0% | 96.5% | 61.4% | 84.2% | 87.7% | 0.860 |
| **+ 重排 `ms-marco-MiniLM-L-6-v2`** | **91.2%** | 96.5% | **98.2%** | **63.2%** | **87.7%** | **89.5%** | **0.939** |

#### 三个和直觉不一样的结论

**① 重排的第一个模型是负作用。**
本来预期"召回 20 → 精排到 3"一定涨，实测 `BAAI/bge-reranker-base`（国内最常被推荐的）
把 Recall@1 从 87.7% 打到 78.9%。

排查思路是先分开两种可能：**是"重排没用"，还是"这个模型不合适"？**
证据指向后者——hybrid 的 Recall@5 已经 98.2%，说明答案基本都在前 5 里，
重排只要不乱动就能保住；它却把 Recall@3 拖到 93.0%，说明它在**主动把正确片段往下排**。

换一个在英文 passage ranking（MS MARCO）上训练的模型后：
**Recall@1 涨 3.5 个百分点，MRR 从 0.924 到 0.939。**

> **结论：重排模型必须按语言和领域选，而且必须实测。**
> `bge-reranker-base` 训练语料以中文为主，放到英文论文上是负作用——
> "cross-encoder" 只是结构，不代表它懂你的数据。

同时如实记下**代价**：Recall@3 从 98.2% 掉到 96.5%。
它换来的是"最相关的那条排得更靠前"，代价是"第 3 名附近的边际召回"变弱。
**这是权衡，不是纯赢。**

**② 为什么用 RRF，不用分数加权？**
向量检索返回余弦距离（0~2，越小越像），BM25 返回无上界的相关性分数（可能 0.3 也可能 30）。
两者量纲完全不同，加权求和要反复调权重且不稳定。
RRF 只用**排名**不用**分数**（`score = Σ 1/(k + rank)`），天然免疫量纲问题。

**③ `chunk_size=2000` 在 AnswerHit@5 上反超（82.5%）。**
片段越大越容易"顺带"把答案原文包进去，但它 Recall@1 只有 54.4%——
说明大片段会命中正确页面却排不靠前。
这正好印证 **Recall@k 和 AnswerHit@k 测的不是同一件事**，只看一个会得出错误结论。

### 5.4 多跳 Agent 的实际运行

用跨文档问题考察（`Compare the pretraining objectives of BERT and GPT-3.`）：

| 轮 | 撒网 k | 本轮新收 | 累计引用 | judge | 下一轮 query |
|---|---:|---:|---:|---|---|
| 1 | 3 | 3 | 3 | ❌ 不够 | `BERT masked language model pretraining objective` |
| 2 | 6 | 6 | 9 | ❌ 不够 | `GPT-3 autoregressive language modeling pretraining objective` |
| 3 | 9 | 9 | 18 | ✅ 够了 | — |

第 1 轮只捞到 BERT，judge 自己意识到"缺 GPT-3"并生成指向 GPT-3 的检索词。
撒网范围每轮放宽（`k = 3 × 轮数`），累计引用 3 → 9 → 18；证据按 `chunk_id` 去重，
同一片段不会被收两次。

这里最能体现"Agent 而不是固定 pipeline"的一点：**下一轮的检索词是 judge 现场写的，
不是从原问题里抽的**。它先判断"手上缺什么"，再把这个判断直接翻译成检索词。

> 同一份代码、只换一个不跨文档的问题，就只跑 1 轮（见 §7.3 ①）。
> 判断多跳是否真的在工作，必须用跨文档问题测。

### 5.5 性能：时间花在哪

接入 Langfuse 后拿到一次完整链路的拆解（上面那道题，3 轮）：

| 环节 | 调用次数 | 耗时 | 占比 |
|---|---:|---:|---:|
| 检索 + 重排 | 3 | 1.625 s | 23% |
| judge 判定（"够了吗"） | 3 | 2.490 s | 36% |
| done 最终作答 | 1 | 2.843 s | 41% |
| **整条链路** | — | **6.979 s** | 100% |

**瓶颈在大模型，不在检索。** 3 轮检索加重排一共才 1.6 秒，一次最终作答就要 2.8 秒；
大模型两项合计 5.3 秒，占了 **76%**。

所以优化顺序很清楚：

- 想缩短总时长 → 先减少 LLM 调用次数（judge 是否每次都必要、能否对简单问题快速放行），
  而不是去调 `chunk_size`、换重排模型这些检索参数；
- 检索层该优化的是**质量**（Recall@k），不是速度——它已经不是瓶颈了。

---

## 6. 复现流程

### 6.1 环境准备

**1) 安装 Ollama 并下载模型**（检索侧的向量化必须用本地 Ollama）：

```bash
ollama pull nomic-embed-text    # 向量模型
ollama pull qwen2.5:3b          # 本地生成模型（走 DeepSeek 时用不到，但建议留着）
```

> 如果模型不在默认目录（如 `E:\Ollama\models`），**命令行启动必须显式带上**——
> `ollama serve` 不会自动继承这个环境变量：
> ```powershell
> $env:OLLAMA_MODELS = "E:\Ollama\models"; ollama serve
> ```

**2) Python 环境**：

```bash
python -m venv venv
venv\Scripts\activate          # PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **装依赖很慢就换国内源。** `torch` 的 wheel 有 124 MB，
> 实测官方源只有 191 KB/s（要 11 分钟），阿里云镜像有 800 KB/s（约 2.6 分钟）：
> ```bash
> pip install -i https://mirrors.aliyun.com/pypi/simple/ torch sentence-transformers
> ```

> 终端中文乱码就先 `chcp 65001` 切 UTF-8 代码页。

### 6.2 配置密钥

在 **`config.py` 同目录**建 `.env`（已 gitignore）：

```
# 走云端大模型时需要（config.py 里 LLM_PROVIDER = "deepseek"）
DEEPSEEK_API_KEY=你的key

# 可选：把每次提问的完整轨迹上报 Langfuse（不填则自动跳过，不影响运行）
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

`config.py` 里的 `LLM_PROVIDER` 决定走哪家：

```python
LLM_PROVIDER = "deepseek"   # "ollama" -> 本地 qwen2.5:3b；"deepseek" -> 云端 API
```

三条约定：**密钥不进代码**（只有 `os.getenv`，无硬编码）、
**不进仓库**（`.env` 已 gitignore）、**读不到就报错**（直接 `RuntimeError`，
而不是带着空 key 发请求收一个没信息量的 401）。

> 选 `ollama` 走本地时**完全不需要** `.env`，改 `LLM_PROVIDER` 一行即可。
> 注意 `load_dotenv` 默认 `override=False`——**系统/用户环境变量优先于 `.env`**。

**Langfuse 的两条关键设计**：

- **密钥缺失是软失败**。没配 `.env` 的人 clone 下来直接能跑：SDK 只打印一句
  `Client will be disabled` 就自己关掉，服务照常工作，只是不上报。上报本身走后台线程，
  网络异常也只留一行日志，**不会把请求拖垮**。
- **一行开关**。想彻底关掉，`.env` 里加 `LANGFUSE_TRACING_ENABLED=false`。


### 6.3 建库

```bash
# 建多论文库（Agent 用）—— 最常用
python ingest.py --corpus multi --chunk-size 500 --rebuild

# 建单论文库（评测基线，一般不用动）
python ingest.py --corpus single --chunk-size 500 --rebuild

# 一次建好三组 chunk_size（消融实验用）
python ingest.py --chunk-size 500 1000 2000 --rebuild
```

**两种语料分工不同，都要留着**：

| 集合 | 片段数 | 用途 |
|---|---:|---|
| `attention_paper_500` | 129 | 单论文，**评测基线**（§5.2/5.3 的数字来自它） |
| `papers_500` | 1687 | 多论文，**给 Agent 用** |

> 不能合并成一个：`gold_qa.json` 那 57 题**全是 Attention 的**，
> 拿它去打 5 篇论文的库，指标全是垃圾。

> `--rebuild` 会先删除同名集合；**不传则是在已有数据上追加**，
> 同一个库跑两次向量数会翻倍、检索结果出现大量重复。

### 6.4 运行

**方式一：起服务（推荐，能直接看到逐字输出）**

```bash
uvicorn api:app --reload
```

然后浏览器打开 <http://127.0.0.1:8000>，输入问题即可。
页面会实时吐出进度（第几轮、捞到几条）和答案正文，跑完再补上引用来源。

```bash
# 想用 curl 看裸的 SSE 帧（便于对接自己的程序）
curl -N -X POST http://127.0.0.1:8000/ask/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "Compare the pretraining objectives of BERT and GPT-3."}'
```

其他接口：`POST /ask`（非流式，一次返回完整 JSON，适合程序调用）、
`GET /health`（自检）。

**方式二：命令行**

```bash
python agent/graph.py     # 多跳 Agent
python RAG_v2.py          # 多轮 RAG
python RAG_v1.py          # 单轮 RAG
```

> **演示时请用跨文档问题**（如 `Compare the pretraining objectives of BERT and GPT-3.`）。
> `graph.py` 的 `__main__` 里如果放单文档问题，跑出来只会是 1 跳——
> 看不出多跳的价值（原因见 §7.3）。

### 6.5 评测

```bash
# 对比三种检索策略
python eval/run_eval.py --chunk-size 500 --mode dense hybrid hybrid_rerank

# 对比三个 chunk_size
python eval/run_eval.py --chunk-size 500 1000 2000

# 看哪些题没检索到
python eval/run_eval.py --chunk-size 500 --mode hybrid --detail

# 对比不同重排模型
python eval/rerank_compare.py --model cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 7. 知识点、踩坑与注意事项

> 这一节是整个项目最有价值的部分。下面每一条都实际踩过，
> 记录了**症状 → 排查 → 根因 → 修法**。完整过程见 `docs/` 里的逐日记录。

### 7.1 ⭐ 一类高频 bug：「加了个东西，但没接到该接的地方」

项目里同一个模式反复出现，值得单独拎出来当一类问题看：

| # | 加的东西 | 没接上哪里 | 后果 |
|---|---|---|---|
| 1 | `config.LLM_PROVIDER` | 没有任何代码读它 | 开关形同虚设 |
| 2 | `ingest_one(chunk_size)` 的参数 | 函数体里用了硬编码常量 | `--chunk-size 1000` 会**静默覆盖** 500 的库 |
| 3 | `argparse` 的 `--corpus` | 没传给 `ingest_one()` | `--corpus single` 被静默忽略 |
| 4 | `State.tried` 已问列表 | `PROMPT_JUDGE` 模板里没有 `{tried}` 占位符，`.format()` 也没传 | 大模型**从头到尾没见过它**，复读一点没修 |

**为什么这类 bug 特别危险**：它们**不报错**。
代码能跑、日志正常、结果看起来对——但功能**根本没生效**。

**怎么防**：
- 加一个参数/字段后，**立刻搜一遍它在哪被读**（`grep` 一下变量名）。
- 涉及"模板 + 填模板"两处时，**当成一个不可分割的改动**：
  模板加了 `{x}` 而 `.format()` 没传 → `KeyError`；
  传了而模板没有 → **不报错但白传**（更危险）。
- 写完**用日志验证它真的生效了**，而不是"没报错应该就对了吧"。

### 7.2 检索层

**① `chunk_size` 的选择必须靠实验，不能拍脑袋。**
原来代码硬编码 `chunk_size=1000`，实测是三个候选里最差的（Recall@1 57.9% vs 500 的 75.4%）。
这条结论只能靠评测得到——这就是做评测的意义。

**② 混合检索里，两路分数的量纲不同。**
向量返回余弦距离（有界），BM25 返回无上界相关性分数。**不能直接加权求和**，
用 RRF 只用排名不用分数，天然免疫量纲问题。

**③ 专有名词和数字，向量模型是不敏感的。**
`BLEU`、`WMT 2014`、`16,384` 在 embedding 训练语料里见得少，
关键词精确匹配（BM25）反而一抓一个准。两路互补，融合后 Recall@3 从 91.2% → 98.2%。

**④ 重排模型必须按语言/领域选，还要实测。**
同一个思路，换模型得到相反结论（详见 §5.3 ①）。
**任何"常见做法"都要在自家数据上验证一遍，不能因为它是"标准做法"就默认有效。**

**⑤ 引用页码要 +1。**
PyMuPDF 的 `metadata["page"]` 是 **0 起算**的，直接显示会得到 `p.0`。
展示时要 `page + 1` 才对得上人翻的页码。
（评测内部仍用 0 起算，两者别混。）

### 7.3 ⭐ Agent 层

**① 多跳是否触发，取决于问题是否跨文档——这是最反直觉的一条。**

同一份代码，只换问题：

| 问题 | hops |
|---|---:|
| `What is the BLEU score of the big model on WMT 2014...?` | **1** |
| `Compare the pretraining objectives of BERT and GPT-3.` | **3** |

BLEU 那道题是 Attention 一篇独有的数据，第 1 轮就命中，judge 说"够了" → 收工。

> **所以：语料只有一篇论文时，多跳代码 100% 正确也永远只跑一轮。**
> 这是**环境的锅，不是代码的锅**。
> 判断一个多跳 Agent 是否真在工作，**必须用跨文档问题去测**。

**踩过的坑**：`graph.py` 的 `__main__` 里硬编码的演示题是单跳题，
自己跑一下看到 `hops = 1`，一度以为 Agent 坏了。

**② judge 会「复读」——因为它看不到自己问过什么。**
最初的 `node_judge` 只能看到【原问题】+【已收证据】，
所以判断"还缺 X"时会**再一次生成几乎一样的检索词**，原地打转：
问题没被回答，轮数却在涨，最后只有 `MAX_HOPS` 能拦住它。
**根因**：`State.tried`（已问列表）字段早就加了，但没接进 judge 的 prompt——
加了个东西，没接到该接的地方（§7.1）。
**修法**：把已问过的词喂进 prompt，让模型看见自己走过哪些路，
下一轮才会往**没搜过的方向**问。

**③ `enough` 的语义必须是「可以作答了」，而不是「答案找到了」。**
prompt 原先是"证据够不够回答问题"，模型理解成"答案找到了吗"。
于是对"库里根本没有"的问题，它永远说"没找到" → 永远 `False` → 一路撞到 `MAX_HOPS`。

正确语义是：**"资料里没有"也是一个诚实的回答**，所以确认找不到之后也该判 `enough = true`。

**④ 日志不能说谎。**
`node_done` 原先无条件打印"够了，可以答了"，
但 judge 可能每一轮都判"不够"，只是**被 `MAX_HOPS` 强制刹车**。
必须区分"真够了"和"跳满上限了"，否则日志会掩盖问题。
现在两种结局各打印各的，看日志就知道是哪种。

**⑤ `temperature=0` 让 Agent 可复现。**
`temperature=1`（默认）时每轮都按概率抽样，同一问题两次跑出不同轮数。
**判断类任务不该抖**，judge 用 `temperature=0` 后，两次运行**逐字一致**。
（服务端因批处理，不保证 100% 逐字一致；要绝对可复现得用本地 Ollama。）

**⑥ 昂贵的东西要缓存。**
多跳每一轮都要检索，但 BM25 索引和重排模型**只需要建一次**。
`agent/engine.py` 用模块级变量做了进程级缓存，否则每轮都重新加载模型。

**收敛效果**（同一个"库里根本没答案"的问题）：

```text
优化前：5 轮 / 20 条 citations —— 一路跳到 MAX_HOPS，白等五轮
优化后：2 轮 /  9 条 citations —— 连续两轮没捞到有效证据就停
```

关键在于给图一个"这轮是不是白跑了"的判据：**只看这轮捞到的最高分 `top_score`**——
一条新片段都没捞到，或者捞到了但最高分低于 `MIN_RERANK_SCORE`，都算同一件事：
**这轮没拿到新的有效证据**（`weak_rounds += 1`）。连续 `MAX_WEAK_ROUNDS = 2` 轮就收工。

两个参数不能拍脑袋，**它们是两个独立的问题**：

- **阈值定多少**（"这轮算不算白跑"）：第一轮的检索词就是用户原话，
  中文问题第一轮分数天然最低（实测 `−0.67`），阈值必须低于它，否则中文问题第一轮就被误杀。
- **连续几轮才停**（"白跑几次才放弃"）：定 1 轮同样会杀掉中文问题。
  一次低分是"第一轮的结构性劣势"，**两次连续低分才是真的搜不到**。

把这两个问题混成一个参数，出了误杀就说不清该改哪一半。

### 7.4 ⭐ 环境层

**① Ollama 的 embedding 批量上限是 311 条，而报错信息是假的。**（最坑的一个）

**症状**：建 1687 个片段的库时挂掉，报的是

```text
Post "http://127.0.0.1:52589/tokenize": dial tcp 127.0.0.1:52589:
connectex: No connection could be made because the target machine
actively refused it. (status code: 400)
```

**看起来完全像网络/端口问题**，白查了很久（端口在监听、`curl` 返回 200、单条
`embed_query` 还成功）。

**排查**：不猜了，按批量大小做梯度测试：

```text
   1 OK   8 OK   32 OK   64 OK   128 OK   256 OK   512 FAIL
```

二分得到：**安全上限 311 条/批。**

**根因**：Ollama 收到一批要逐条找后端进程做 `/tokenize`，
几百条把后端连接队列冲垮；而 **Ollama 把这个"连不上自家后端"的错误原样透传**，
于是客户端看到的是一句**和真实原因毫无关系**的"端口拒绝连接"。

**为什么 1687 条会一次性发过去**：看 `langchain_chroma` 源码，
它用 chromadb 的 `get_max_batch_size()` 切批——**那是 SQLite 的变量上限（几千）**。
**它根本不知道我们把 embedding 后端换成了 Ollama。**

**修法**：`config.EMBED_BATCH_SIZE = 100`，在 `create_vector_store` 里自己分批：

```python
for start in range(0, total, step):
    batch     = documents[start : start + step]
    batch_ids = ids[start : start + step] if ids else None   # ← ids 必须同步切！
    if db is None:
        db = Chroma.from_documents(documents=batch, ..., ids=batch_ids, ...)  # 建集合
    else:
        db.add_documents(batch, ids=batch_ids)                              # 追加
```

**教训**：
1. **报错的第一句话可能是假的。** 要用实验去证伪，别被错误信息牵着走。
2. **换一个环节的后端，要检查上一个环节的默认参数还成不成立。**
3. 长任务要**打印进度**，不能静默。

**② `OLLAMA_HOST=0.0.0.0` 会让客户端连不上。**

`0.0.0.0` 是**服务端监听地址**（监听所有网卡），不是**客户端连接地址**。
ollama 的 Python 客户端读了它去连接，在 Windows 上直接
`ConnectError [WinError 10049] 地址无效`，而客户端还会**把底层异常吞掉**，
统一报一句没信息量的 "Failed to connect to Ollama"。

本项目已规避：`config.OLLAMA_BASE_URL` 显式写死 `http://127.0.0.1:11434`。

**③ 报 `status code: 502` 时，根因通常还是 Ollama 没启动。**
系统代理（Clash 监听 `127.0.0.1:7897`）把发往 localhost 的请求也转发走了，
代理连不上后端就回 502。**不要被这个错误码带偏。**

**④ 重排模型下载卡死 → 走镜像站。**
国内直连 huggingface.co 实测 **12 秒无任何响应**（不是慢，是连不上），
换 `https://hf-mirror.com` **0.6s 返回**。
注意 `huggingface_hub` 在 **import 时**就把 `HF_ENDPOINT` 读成了常量，
所以环境变量必须在导入它之前设好——本项目放在 `config.py` 顶层。

**⑤ 中文控制台会 `UnicodeEncodeError`。**
GPT-3 第一页有个 `∗`（U+2217），Windows 的 GBK 控制台打印它直接崩。
**加 `PYTHONIOENCODING=utf-8` 即可**，这是**控制台问题，不是数据问题**。

### 7.5 生成层 / 配置层

**① `.env` 不是环境变量，`load_dotenv()` 必须被调用。**
`os.getenv()` 读的是**进程的环境变量**，不是那个文件。
只 `import` 进来是没用的——这和"函数定义了但没人调用"是同一类问题。

**② OpenAI 系模型的 JSON 模式跟 Ollama 不一样。**

| | 怎么写 |
|---|---|
| Ollama | `ChatOllama(..., format="json")` |
| OpenAI 系 / DeepSeek | `model_kwargs={"response_format": {"type": "json_object"}}` |

而且 OpenAI 系有个**很容易踩的附加条件**：
**prompt 里必须出现 "json" 这个词本身**，否则 API 直接报错。
`src/llm.py` 做了归一化（调用方统一传 `format="json"`，工厂内部转换），
但 **prompt 里那句"只输出一个 JSON 对象"不能省**。

**③ 「判断力不足」是模型能力问题，不是 prompt 工程问题。**
做过一次**控制变量实验**：在**完全相同的证据**下，
本地 `qwen2.5:3b` **10 次全部**判"证据不够"，DeepSeek 第 1 轮就判对。
→ 换更强的模型比反复调 prompt 有效得多。

**④ 判断类任务用 `temperature=0`**（理由见 §7.3 ⑤）。

### 7.6 ⭐ 服务层与可观测性

**① SSE 的帧边界是那个空行，不是 `data:`。**

```text
data: {"type": "progress", "round": 1, "hits": 3}
                                                ←← 这个空行才是"一帧结束"
data: {"type": "final", ...}
```

浏览器端 `ReadableStream` 收到的是**字节流，不是消息流**，TCP 想怎么切就怎么切。
一个 JSON 被拆成两半、或者两帧挤在一次 `read()` 里，都完全正常。
所以前端必须**自己攒 buffer、按空行切、切不完整就留着**——
`static/index.html` 里那段 `buf.split("\n\n")` + `buf.pop()` 干的就是这件事。

**② 中文用户专属：全角引号导致 422。**

中文输入法下打出来的是全角 `“`，屏幕上和半角 `"` 几乎一样，电脑却认为是完全不同的字符，
JSON 直接解析失败 → HTTP **422**。字符串**内部**的中文、全角问号都没问题，
**只有结构位置上的符号**必须是半角。

这也是 422 和 500 值得分清的地方：**422 说明请求在门口就被拦下了，
业务函数一行都没跑**。看到 422 就不必去翻图或检索的代码。

**③ 前端渲染：先转义，再套标记——顺序本身就是安全边界。**

模型输出是要当 HTML 渲染的（要支持加粗/列表），所以必须先把 `<` `>` `&` 转义，
再把我们自己拼的标签加进去。模型就算吐出 `<script>`，转义后也只是一串纯文字。
转义内部**`&` 必须第一个替换**，否则会把后面生成的 `&lt;` 再转一次，页面上冒出半截符号。

**④ token 流和 Langfuse 挂在同一个槽上。**

两者都走 LangChain 的 callback 机制：一个是 `stream_mode="messages"`，
一个是 `config["callbacks"]`。所以**接可观测性时 `agent/` 一行都没改**——
API 层加一个 handler 就够了。这正是把横切关注点放到框架机制上、而不是塞进业务代码的好处。

**⑤ 可观测性必须是软失败。**

Langfuse 上报走后台线程，密钥缺失就自己关掉、网络异常只留一行日志，
**既不阻塞请求也不影响答案**。实测把密钥改错：服务照常 200、答案照常正确，
只是不再有 trace。观测是用来**帮**你的，不该变成新的故障点。

---

## 8. 工作记录

`docs/` 下是逐日的工作记录，写的是**过程和判断依据**，不是结论摘要——
每一条参数（阈值、上限、分块大小）背后是**哪次实验、哪个数据**把它定下来的。

- [2026-10-07](docs/worklog-2026-10-07.md) — chunk_size 消融、混合检索、重排选型
- [2026-10-08](docs/worklog-2026-10-08.md) — LangGraph 入门、`MAX_HOPS` 刹车、评测可复现性的边界
- [2026-10-09](docs/worklog-2026-10-09.md) — 接上真实检索与 LLM 判断、"判断力"控制变量实验
- [2026-10-10](docs/worklog-2026-10-10.md) — 接入 DeepSeek；多论文语料；**多跳首次真正跑通**；修复读与可复现性
- [2026-10-12](docs/worklog-2026-10-12.md) — **给 Agent 加防雷 + 服务化 Day 1**：JSON 解析容错、检索降级；FastAPI 服务、lifespan 预热、citations 结构化
- [2026-10-13](docs/worklog-2026-10-13.md) — **防雷二：让"检索到了但都不相关"的问题早点停**；阈值 0 的依据是 raw logit 决策边界，上限 2 的依据是中文第一轮结构性低分
- [2026-10-14](docs/worklog-2026-10-14.md) — **服务化 Day 2：SSE 流式 + 单页前端 + 可观测性**；按节点粒度推进度、答案逐字吐出（`agent/` 一行未改）、接入 Langfuse 后**首次量化出 76% 的时间在等大模型**

