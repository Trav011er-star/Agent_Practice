# Paper-Agent

**一个带多跳检索 Agent 的论文问答助手。** 把 PDF 论文建成向量库，混合同检索召回证据，
再由 **LangGraph 多跳 Agent** 判断"证据够不够"，不够就自己改写检索词再搜一轮，
最后生成**每句都带出处**的答案。

两层都做了**可复现的量化评测**：检索层 57 条标注问题测 Recall@k / MRR；
Agent 层用跨文档问题验证多跳是否真的触发。

---

## 目录

- [1. 项目简介](#1-项目简介)
  - [解决什么问题](#解决什么问题) · [一句话亮点](#一句话亮点) · [语料](#语料)
- [2. 核心功能](#2-核心功能)
- [3. 技术栈](#3-技术栈)
- [4. 系统架构](#4-系统架构)
  - [4.1 建库（离线，跑一次）](#41-建库离线跑一次) · [4.2 多跳 Agent（在线）](#42-多跳-agent在线) · [4.3 目录结构](#43-目录结构)
- [5. 效果数据](#5-效果数据)
  - [5.1 指标定义](#51-指标定义) · [5.2 chunk_size 消融实验](#52-chunk_size-消融实验) · [5.3 检索策略消融实验](#53-检索策略消融实验) · [5.4 多跳 Agent 的验证](#54-多跳-agent-的验证定性暂无量化)
- [6. 复现流程](#6-复现流程)
  - [6.1 环境准备](#61-环境准备) · [6.2 配置密钥](#62-配置密钥只有用-deepseek-才需要) · [6.3 建库](#63-建库) · [6.4 运行](#64-运行) · [6.5 评测](#65-评测)
- [**7. 知识点、踩坑与注意事项**](#7-知识点踩坑与注意事项)
  - [7.1 最高频的一类 bug：「加了个东西，但没接到该接的地方」](#71--最高频的一类-bug加了个东西但没接到该接的地方)
  - [7.2 检索层](#72-检索层) · [7.3 Agent 层](#73--agent-层) · [7.4 环境层](#74--环境层) · [7.5 生成层 / 配置层](#75-生成层--配置层)
- [8. 工作记录](#8-工作记录)

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

---

## 3. 技术栈

| 层 | 用了什么 | 为什么 |
|---|---|---|
| 编排 | **LangGraph** | 多跳循环是**有状态、带条件跳转**的流程，用图来表达比写 `while` 循环清晰得多 |
| 检索 | 自实现 **BM25** + 向量检索 + **RRF** 融合 | 专有名词/数字（`BLEU`、`WMT 2014`、`16,384`）向量模型不敏感，关键词匹配一抓一个准 |
| 重排 | **sentence-transformers** cross-encoder | 从 20 条候选里精挑 3 条 |
| 向量模型 | Ollama **`nomic-embed-text`**（本地，768 维） | 免 API、可离线 |
| 生成模型 | **DeepSeek**（云端）/ Ollama **`qwen2.5:3b`**（本地） | 可切换；实测判断力差距很大，见 §7.3 |
| 向量库 | **Chroma**（本地持久化） | 轻量、免部署 |
| 解析/分块 | **PyMuPDF** + langchain `RecursiveCharacterTextSplitter` | 按页解析，保证 chunk 的页码唯一 |
| 评测 | 自实现 Recall@k / AnswerHit@k / MRR | 指标定义见 §5.1 |
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

### 4.3 目录结构

```text
paper-Agent/
├── config.py              # 所有可调参数集中在此（含 LLM_PROVIDER 开关）
├── ingest.py              # 建库：PDF -> 分块 -> 向量化 -> Chroma
├── RAG_v1.py / RAG_v2.py  # 单轮 / 多轮 RAG
├── requirements.txt
├── .env                   # 密钥（已 gitignore，需自行创建，见 6.2）
├── agent/
│   ├── graph.py           # LangGraph 多跳 Agent：collect / judge / done
│   ├── engine.py          # 检索引擎的进程级缓存
│   └── graph_demo.py      # 早期"假证据"版骨架（学习过程存档，非有效代码）
├── src/
│   ├── loader.py          # PDF 解析
│   ├── splitter.py        # 分块
│   ├── embedding.py       # 向量化 + Chroma 读写（含分批写入）
│   ├── llm.py             # 大模型工厂（provider 切换）
│   ├── retrieval.py       # 混合检索与重排
│   └── health.py          # 启动自检：Ollama 是否可用、模型是否已下载
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

### 5.4 多跳 Agent 的验证（定性，暂无量化）

用跨文档问题考察（`Compare the pretraining objectives of BERT and GPT-3.`）：

| 轮 | 撒网 k | 新收 | judge | 下一轮 query |
|---|---:|---:|---|---|
| 1 | 3 | 3 | ❌ 不够 | `GPT-3 pretraining objective autoregressive language modeling` |
| 2 | 6 | 6 | ❌ 不够 | `BERT masked language model next sentence prediction pretraining objectives` |
| 3 | 9 | 9 | ❌ 不够 | `GPT-3 autoregressive language model pretraining objective next token prediction` |
| 4 | 12 | 5 | ✅ 够了 | — |

第 1 轮只捞到 BERT，judge 自己意识到"缺 GPT-3"并生成指向 GPT-3 的检索词。
第 4 轮 k=12 只新收 5 条，说明 `chunk_id` 去重生效。

> ⚠️ **Agent 层目前没有量化评测**，只有上面这种定性验证。
> 检索层有 57 题的完整指标，Agent 层还没有对应的评测集。

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

### 6.2 配置密钥（只有用 DeepSeek 才需要）

`config.py` 里的 `LLM_PROVIDER` 决定走哪家：

```python
LLM_PROVIDER = "deepseek"   # "ollama" -> 本地 qwen2.5:3b；"deepseek" -> 云端 API
```

选 `deepseek` 时，在 **`config.py` 同目录**建 `.env`：

```
DEEPSEEK_API_KEY=你的key
```

三条约定：**密钥不进代码**（只有 `os.getenv`，无硬编码）、
**不进仓库**（`.env` 已 gitignore）、**读不到就报错**（直接 `RuntimeError`，
而不是带着空 key 发请求收一个没信息量的 401）。

> 选 `ollama` 走本地时**完全不需要** `.env`，改 `LLM_PROVIDER` 一行即可。
> 注意 `load_dotenv` 默认 `override=False`——**系统/用户环境变量优先于 `.env`**。

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

### 7.1 ⭐ 最高频的一类 bug：「加了个东西，但没接到该接的地方」

这个模式在项目里**犯了 4 次**，值得单独拎出来：

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
| `Compare the pretraining objectives of BERT and GPT-3.` | **4** |

BLEU 那道题是 Attention 一篇独有的数据，第 1 轮就命中，judge 说"够了" → 收工。

> **所以：语料只有一篇论文时，多跳代码 100% 正确也永远只跑一轮。**
> 这是**环境的锅，不是代码的锅**。
> 判断一个多跳 Agent 是否真在工作，**必须用跨文档问题去测**。

**踩过的坑**：`graph.py` 的 `__main__` 里硬编码的演示题是单跳题，
自己跑一下看到 `hops = 1`，一度以为 Agent 坏了。

**② judge 会「复读」——因为它看不到自己问过什么。**
最初的 `node_judge` 只能看到【原问题】+【已收证据】，
所以判断"还缺 X"时会**再一次生成几乎一样的检索词**，原地打转。
一次压测跑出 **10 轮全程复读**（第 6 轮和第 5 轮一字不差）。
**修法**：加 `tried: list[str]` 记录已问过的词并喂进 prompt。

**③ `enough` 的语义必须是「可以作答了」，而不是「答案找到了」。**
prompt 原先是"证据够不够回答问题"，模型理解成"答案找到了吗"。
于是对"库里根本没有"的问题，它永远说"没找到" → 永远 `False` → 一路撞到 `MAX_HOPS`。

正确语义是：**"资料里没有"也是一个诚实的回答**，所以确认找不到之后也该判 `enough = true`。

**④ 日志不能说谎。**
`node_done` 原先无条件打印"够了，可以答了"，
但 judge 可能 10 轮全判"不够"，只是**被 `MAX_HOPS` 强制刹车**。
必须区分"真够了"和"跳满上限了"，否则日志会掩盖问题。

**⑤ `temperature=0` 让 Agent 可复现。**
`temperature=1`（默认）时每轮都按概率抽样，同一问题两次跑出 4 轮 / 5 轮不同结果。
**判断类任务不该抖**，judge 用 `temperature=0` 后，两次运行**逐字一致**。
（诚实补充：服务端因批处理，不保证 100% 逐字一致；要绝对可复现得用本地 Ollama。）

**⑥ 昂贵的东西要缓存。**
多跳每一轮都要检索，但 BM25 索引和重排模型**只需要建一次**。
`agent/engine.py` 用模块级变量做了进程级缓存，否则每轮都重新加载模型。

**收敛效果**（同一个"库里没答案"的问题，逐步优化）：

```text
10 轮  →  5 轮  →  4 轮  →  3 轮
```

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

---

## 8. 工作记录

`docs/` 下是逐日的工作记录，写的是**过程和踩坑**，不是结论摘要——
包括方案被推翻、实验做错、以及自己的假设后来站不住的地方。

- [2026-10-07](docs/worklog-2026-10-07.md) — chunk_size 消融、混合检索、重排选型
- [2026-10-08](docs/worklog-2026-10-08.md) — LangGraph 入门、`MAX_HOPS` 刹车、评测可复现性的边界
- [2026-10-09](docs/worklog-2026-10-09.md) — 接上真实检索与 LLM 判断、"判断力"控制变量实验
- [2026-10-10](docs/worklog-2026-10-10.md) — 接入 DeepSeek；多论文语料；**多跳首次真正跑通**；修复读与可复现性

> ⚠️ **已知的诚实性说明**：
> - §5.2 的 `Recall@1` 会跨进程抖动（已加前提说明），主结论请看 Recall@3 / MRR。
> - §5.3 的重排**不是全面胜出**，Recall@3 反而降了 1.7 个百分点。
> - §5.4 的 Agent 只有**定性验证**，**没有量化评测集**。

