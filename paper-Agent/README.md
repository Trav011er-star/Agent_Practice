# Paper-Agent

基于 RAG 的论文阅读助手：把 PDF 论文建成可检索的向量库，支持多轮问答，并且**用一套可复现的评测集量化检索质量**。

在检索之上搭了一层 **LangGraph 多跳 Agent**：由大模型自己判断"证据够不够"，
不够就改写检索词再搜一轮，够了就生成**带出处引用**的答案。
生成模型支持**本地 Ollama / 云端 DeepSeek 一键切换**。

当前聚焦单篇论文（*Attention Is All You Need*）打通全链路，后续扩展为多论文文献助手。

---

## 1. 已完成的能力

| 环节 | 实现 | 位置 |
|---|---|---|
| 加载 | PyMuPDFLoader 按页解析 PDF | `src/loader.py` |
| 分块 | RecursiveCharacterTextSplitter，参数可配 | `src/splitter.py` |
| 向量化 | Ollama `nomic-embed-text`（本地） | `src/embedding.py` |
| 存储 | Chroma 本地持久化，按 chunk_size 分集合 | `src/embedding.py` |
| 生成 | 本地 Ollama `qwen2.5:3b` / 云端 DeepSeek，**一键切换** | `src/llm.py` |
| **混合检索** | **自实现 BM25 + 向量检索，RRF 融合** | **`src/retrieval.py`** |
| **重排** | cross-encoder 精排（模型按语言/领域**实测选型**） | `src/retrieval.py` |
| 单轮 RAG | LCEL 管道 `prompt \| llm \| StrOutputParser` | `RAG_v1.py` |
| 多轮 RAG | query rewrite（指代消解）+ 对话历史注入 | `RAG_v2.py` |
| **多跳 Agent** | **LangGraph：检索 → 判断证据是否充分 → 改写 query → 再检索，带 `MAX_HOPS` 刹车** | **`agent/graph.py`** |
| **检索引擎缓存** | 多跳每一轮不再重建 BM25 索引 / 重排模型 | `agent/engine.py` |
| **答案溯源** | 回答内嵌 `[论文名, p.页码]`，可追溯到片段级 `chunk_id` | `agent/graph.py` |
| **检索评测** | **57 条标注问题，Recall@k / AnswerHit@k / MRR** | **`eval/`** |
| 建库脚本 | 支持多 chunk_size 批量建库、可重建 | `ingest.py` |

---

## 2. 目录结构

```
paper-Agent/
├── config.py              # 所有可调参数集中在这里（含 LLM_PROVIDER 全局开关）
├── ingest.py              # 建库：PDF -> 分块 -> 向量化 -> Chroma
├── RAG_v1.py              # 单轮 RAG
├── RAG_v2.py              # 多轮 RAG（含 query rewrite）
├── requirements.txt
├── .env                   # 密钥（已 gitignore，需自行创建，见 3.2）
├── agent/
│   ├── graph.py           # LangGraph 多跳 Agent：collect / judge / done
│   ├── engine.py          # 检索引擎的进程级缓存
│   └── graph_demo.py      # 早期"假证据"版骨架（学习过程存档，非有效代码）
├── docs/                  # 逐日工作记录（含踩坑、实验与结论）
├── src/
│   ├── loader.py          # PDF 解析
│   ├── splitter.py        # 分块
│   ├── embedding.py       # 向量化 + Chroma 读写
│   ├── llm.py             # 大模型工厂（provider 切换）
│   ├── retrieval.py       # 混合检索（BM25 + 向量 + RRF）与重排
│   └── health.py          # 启动自检：Ollama 是否可用、模型是否已下载
├── eval/
│   ├── gold_qa.json       # 评测集：57 条问题 + 标准答案页码 + 原文证据
│   ├── metrics.py         # Recall@k / AnswerHit@k / MRR
│   ├── run_eval.py        # 一键跑评测，输出对比表
│   ├── rerank_compare.py  # 对比不同重排模型（换模型一行命令）
│   └── results/           # 评测结果存档（简历数据来源）
├── data/papers/           # 论文 PDF
└── vectorstore/           # Chroma 数据库（已 gitignore，可重建）
```

---

## 3. 环境准备

### 3.1 安装 Ollama 与 Python 依赖

需要本地安装 [Ollama](https://ollama.com/) 并下载两个模型：

```bash
ollama pull nomic-embed-text    # 向量模型
ollama pull qwen2.5:3b          # 生成模型
```

> **注意**：如果你的模型不在默认目录（例如放在了 `E:\Ollama\models`），
> 需要先设置环境变量 `OLLAMA_MODELS` 再启动 Ollama，否则会报 502 / 找不到模型。
> ```powershell
> $env:OLLAMA_MODELS = "E:\Ollama\models"
> ollama serve
> ```

安装 Python 依赖：

```bash
python -m venv venv
venv\Scripts\activate          # PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 如果终端里中文显示乱码，先执行 `chcp 65001` 切到 UTF-8 代码页。

> **装依赖很慢的话换国内源。** `torch` 的 wheel 有 124 MB，
> 实测官方源只有 **191 KB/s**（要 11 分钟），阿里云镜像有 **800 KB/s**（约 2.6 分钟）：
> ```bash
> pip install -i https://mirrors.aliyun.com/pypi/simple/ torch sentence-transformers
> ```
> 也可以一次性配好，之后都不用带参数：
> ```bash
> pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
> ```

### 3.2 配置密钥（只有用 DeepSeek 才需要）

`config.py` 里的 `LLM_PROVIDER` 决定走哪一家：

```python
LLM_PROVIDER = "deepseek"   # "ollama" -> 本地 qwen2.5:3b；"deepseek" -> 云端 API
```

选 `deepseek` 时，需要在 **`config.py` 同目录**下建一个 `.env` 文件：

```
DEEPSEEK_API_KEY=你的key
```

三条设计约定：

- **密钥不进代码**：代码里只有 `os.getenv("DEEPSEEK_API_KEY")`，没有任何硬编码。
- **密钥不进仓库**：`.env` 已在 `.gitignore` 中，不会被提交。
- **读不到就报错**：拿不到 key 时直接 `RuntimeError`，而不是带着空 key 去发请求
  （否则只会收到一个没信息量的 `401`，白花时间排查）。

> 选 `ollama` 走本地时**完全不需要** `.env`，把 `LLM_PROVIDER` 改成 `"ollama"` 即可。
> 两条路都能跑通全链路，切换只改这一行。

> 注意：`.env` 本身只是一个**文本文件**，是 `load_dotenv()` 把它读成环境变量的。
> 而且 `load_dotenv` 默认 `override=False`——**系统/用户环境变量优先于 `.env`**。

---

## 4. 使用

### 建库

```bash
# 建默认配置的库
python ingest.py

# 建多个 chunk_size 的库（做消融实验）
python ingest.py --chunk-size 500 1000 2000 --rebuild
```

`--rebuild` 会先删除同名集合，避免重复写入（不传则是在已有集合上追加）。

### 评测

```bash
# 对比三种检索策略（最常用）
python eval/run_eval.py --chunk-size 500 --mode dense hybrid hybrid_rerank

# 对比三个 chunk_size
python eval/run_eval.py --chunk-size 500 1000 2000

# 看哪些题目没检索到
python eval/run_eval.py --chunk-size 500 --mode hybrid --detail

# 对比不同重排模型（会先跑一遍 hybrid 作基线）
python eval/rerank_compare.py --chunk-size 500
python eval/rerank_compare.py --model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 问答

```bash
python RAG_v2.py     # 多轮对话
python RAG_v1.py     # 单轮
```

### 多跳 Agent

```bash
python agent/graph.py
```

流程是三个节点：

```text
        ┌─────────────────────────────┐
        ↓                             │ 不够：改写 query
START → collect（检索） → judge（判断够不够）
                              │
                              └─ 够了 → done（生成带出处的答案） → END
```

- `collect`：调 `retrieve()` 取片段，用 `chunk_id` 去重后累加进 `items`
- `judge`：把「问题 + 已有证据」给大模型，要它回一个 JSON
  （`{"enough": bool, "next_query": str}`）
- `done`：把证据给大模型，要求**只依据资料作答**并标注 `[论文名, p.页码]`

`MAX_HOPS` 是刹车，防止 `judge` 一直判"不够"时无限循环。

> 前提：先按 3.2 配好 `.env`（若走 DeepSeek），并确保已建库（`python ingest.py`）。
> `agent/engine.py` 会自动把项目根目录插进 `sys.path`，所以在项目根目录下直接跑即可。

---

## 4.5 常见问题排查

所有入口脚本（`ingest.py` / `eval/run_eval.py` / `RAG_v1.py` / `RAG_v2.py`）
启动时都会先调用 `src/health.py` 做自检，把"报错看不懂"的情况翻译成人话。

> `agent/graph.py` 目前没有做这层自检，但它同样依赖 Ollama 的 **embedding** 服务
> （生成模型虽然换成了 DeepSeek，检索那一侧的向量化仍是本地 `nomic-embed-text`）。
> 所以跑 Agent 之前，**Ollama 仍必须是启动状态**。

下面两个坑都实际踩过，记录在此：

### 坑 1：`OLLAMA_HOST` 被设成了 `0.0.0.0`

**症状**：`python ingest.py` 报
```
ConnectionError: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible.
```
但 `curl http://127.0.0.1:11434/api/tags` 明明返回 200。

**原因**：环境变量 `OLLAMA_HOST=0.0.0.0:11434`。
`0.0.0.0` 是**服务端监听地址**（表示监听所有网卡），不是**客户端连接地址**。
ollama 的 Python 客户端读了这个变量去连接，在 Windows 上直接失败：
```
ConnectError [WinError 10049] 在其上下文中，该请求的地址无效。
```
而且客户端把底层异常吞掉，统一抛出上面那句没有信息量的提示。

**本项目已规避**：`config.OLLAMA_BASE_URL` 显式写死 `http://127.0.0.1:11434`，
不受该环境变量影响。自检发现冲突时也会打印警告。

**建议顺手改掉**（否则其它用 ollama 客户端的工具仍会踩）：
```powershell
setx OLLAMA_HOST "127.0.0.1:11434"     # 改掉
reg delete "HKCU\Environment" /v OLLAMA_HOST /f   # 或直接删掉
```

### 坑 2：模型不在默认目录

**症状**：Ollama 服务能连上，但报"找不到模型"。

**原因**：模型放在 `E:\Ollama\models` 等非默认目录，而命令行启动
`ollama serve` **不会自动继承** `OLLAMA_MODELS` 环境变量（桌面版会）。

**解决**：启动时显式带上：
```powershell
$env:OLLAMA_MODELS = "E:\Ollama\models"; ollama serve
```

### 坑 3：报 `status code: 502`

**原因**：系统代理（如 Clash 监听 `127.0.0.1:7897`）把发往 localhost 的请求
也转发走了，代理连不上后端就回 502。**根因通常还是 Ollama 没启动**，
不要被这个错误码带偏。

### 坑 4：重排模型下载卡死

**症状**：第一次跑 `hybrid_rerank` 时长时间无输出，最后超时。

**原因**：`BAAI/bge-reranker-base` 需要从 huggingface.co 下载，
国内直连实测 **12 秒无任何响应**（不是慢，是根本连不上）。

**解决**：走镜像站。`config.py` 里已经默认设好：
```python
HF_ENDPOINT = "https://hf-mirror.com"   # 实测 0.6s 返回
```
注意 `huggingface_hub` 是在 **import 时**把 `HF_ENDPOINT` 读成常量的，
所以环境变量必须在导入它之前设好——本项目放在 `config.py` 顶层，
任何入口脚本都会先 `import config`，因此一定生效。

### 坑 5：接 DeepSeek 时踩的两个坑

**(a) 忘了调用 `load_dotenv()`**

**症状**：启动就报 `RuntimeError: 没读到 DEEPSEEK_API_KEY`，但 `.env` 明明建好了。

**原因**：`os.getenv()` 读的是**进程的环境变量**，不是那个文件。
`.env` 要变成环境变量，必须有人**调用** `load_dotenv()`——
只 `import` 进来是没用的。这和"函数定义了但没人调用"是同一类问题。

**本项目已处理**：`config.py` 顶层调用 `load_dotenv(PROJECT_ROOT / ".env")`，
且位置在 `os.getenv(...)` **之前**。

**(b) OpenAI 系模型的 JSON 模式跟 Ollama 不一样**

**原因**：两家"要求 JSON 输出"的说法不同——

| | 怎么写 |
|---|---|
| Ollama | `ChatOllama(..., format="json")` |
| OpenAI 系 / DeepSeek | `model_kwargs={"response_format": {"type": "json_object"}}` |

而且 OpenAI 系有个**很容易踩的附加条件**：
**prompt 里必须出现 "json" 这个词本身**，否则 API 直接报错。
所以 `judge` 的 prompt 能用（写了"只输出一个 JSON 对象"），
而一个不含 "json" 字样的 prompt 拿去要 JSON 就会当场失败。

**本项目已处理**：`src/llm.py` 的 `get_llm_model()` 里做了归一化——
调用方统一传 `format="json"`，工厂内部转成各家的写法。
但 **prompt 里那句"只输出一个 JSON 对象"不能省**（见 `PROMPT_JUDGE`）。

---

## 5. 评测结果

### 5.1 指标定义

- **Recall@k**：前 k 个检索片段中，至少有一个来自正确页面的题目占比。
  衡量"该找的东西有没有被捞出来"。
- **AnswerHit@k**：前 k 个片段中**真的包含标准答案原文**的题目占比（子串匹配）。
  比 Recall@k 严格得多——落在正确页码不等于捞到了那句话。
- **MRR@10**：第一个命中片段的排名倒数平均值。越接近 1 说明正确答案排得越靠前。
  因为最终只有前 k 个片段会喂给大模型，排名靠后等于没检索到。

### 5.2 chunk_size 消融实验

评测集：57 条问题（覆盖事实、数字表格、名词定义、公式四类），检索深度 k=10。
模型：`nomic-embed-text`。纯检索评测，不涉及生成。

| chunk_size | 片段数 | Recall@1 | Recall@3 | Recall@5 | AnswerHit@1 | AnswerHit@3 | AnswerHit@5 | MRR@10 |
|-----------:|-------:|---------:|---------:|---------:|------------:|------------:|------------:|-------:|
| **500** | 129 | **75.4%** | **91.2%** | **94.7%** | 50.9% | 73.7% | 77.2% | **0.841** |
| 1000 | 52 | 57.9% | 80.7% | 91.2% | 42.1% | 64.9% | 73.7% | 0.706 |
| 2000 | 27 | 54.4% | 84.2% | 93.0% | 45.6% | **75.4%** | **82.5%** | 0.702 |

> **可复现性的边界（实测）**
>
> 同一 Ollama 进程内连跑两次，结果**逐位一致**。但**换一个 Ollama 进程**之后，
> **Recall@1 会抖约 ±5 个百分点**，而 **Recall@3 / Recall@5 / MRR 完全不动**。
>
> 逐题对比后定位到：变化的题目**全部**是"第 1 名 ↔ 第 2 名互换"
> （RR 1.00 ↔ 0.50）——也就是**检索到的内容没变，抖的只是近似并列时谁排前面**。
> 疑似根因是不同进程的 embedding 浮点差异，但**该假设未经证实**。
>
> 所以本表的主结论请看 **Recall@3 / MRR**，**Recall@1 只作参考**。
> 完整排查过程见 [docs/worklog-2026-10-08.md](docs/worklog-2026-10-08.md) §9。

### 5.3 检索策略消融实验

固定 `chunk_size=500`（5.2 选出的最优配置），对比纯向量检索和混合检索。
混合检索 = 向量检索 + BM25 各召回 20 条，用 RRF（Reciprocal Rank Fusion, k=60）融合。

| 检索策略 | Recall@1 | Recall@3 | Recall@5 | AnswerHit@1 | AnswerHit@3 | AnswerHit@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense（纯向量） | 75.4% | 91.2% | 94.7% | 50.9% | 73.7% | 77.2% | 0.841 |
| hybrid（BM25 + 向量 + RRF） | 87.7% | **98.2%** | 98.2% | 61.4% | 86.0% | 89.5% | 0.924 |
| + 重排 `bge-reranker-base` | 78.9% | 93.0% | 96.5% | 61.4% | 84.2% | 87.7% | 0.860 |
| **+ 重排 `ms-marco-MiniLM-L-6-v2`** | **91.2%** | 96.5% | **98.2%** | **63.2%** | **87.7%** | **89.5%** | **0.939** |

#### 重排这一项，结果和直觉不一样

本来预期"召回 20 条 → cross-encoder 精排到 3 条"一定涨，**实测第一个模型是负作用**：
`BAAI/bge-reranker-base`（国内最常被推荐的那个）把 Recall@1 从 87.7% 打到了 78.9%。

排查思路是先把两种可能性分开：**是"重排这个思路没用"，还是"这个模型不合适"？**

证据指向后者——hybrid 的 Recall@5 已经 98.2%，说明正确答案基本都在融合结果的前 5 里，
重排只要不乱动就能保住这个水平。它却把 Recall@3 拖到 93.0%，说明它在**主动把正确片段往下排**。

于是换一个在英文 passage ranking（MS MARCO）上训练的模型重测
（`eval/rerank_compare.py`，一行命令切换模型）：

| 重排模型 | Recall@1 | Recall@3 | AnswerHit@3 | MRR@10 |
|---|---:|---:|---:|---:|
| 不重排（hybrid） | 87.7% | **98.2%** | 86.0% | 0.924 |
| `BAAI/bge-reranker-base` | 78.9% | 93.0% | 84.2% | 0.860 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | **91.2%** | 96.5% | **87.7%** | **0.939** |

**结论：重排模型必须按语言和领域选，而且必须实测。**
`bge-reranker-base` 训练语料以中文为主，放到英文论文上是负作用——
"cross-encoder" 只是结构，不代表它懂你的数据。

换成对口的模型后，**Recall@1 涨 3.5 个百分点，MRR 从 0.924 到 0.939**。

同时如实记下它的**代价**：`Recall@3` 从 98.2% 掉到 96.5%。
因为最终只有 top-3 喂给大模型，这个点不能只报喜——
它换来的是"最相关的那条排得更靠前"（Recall@1 / AnswerHit@1 提升），
代价是"第 3 名附近的边际召回"变弱。**这是权衡，不是纯赢。**

**为什么不做"分数加权求和"而是 RRF？**
向量检索返回的是余弦距离（0~2，越小越像），BM25 返回的是无上界的相关性分数（可能 0.3 也可能 30）。
两者量纲完全不同，直接加权求和需要反复调权重且不稳定。
RRF 只用**排名**不用**分数**：`score = Σ 1/(k + rank)`，
天然免疫量纲问题，也不需要调参——这也是它在工业界检索系统里流行的原因。

### 5.4 结论

1. **`chunk_size=500` 整体最优**：Recall@1 比 1000 高 **17.5 个百分点**，
   Recall@3 高 10.5 个百分点，MRR 从 0.706 提升到 **0.841**。
   说明这篇论文里"段落级"的语义单元更接近 500 字，切成 1000 字会把多个主题混在一起，
   稀释掉向量表示的区分度。
2. **`chunk_size=2000` 在 AnswerHit@5 上反超（82.5%）**：
   片段越大，越容易"顺带"把答案原文包进去。
   但它 Recall@1 只有 54.4%，说明大片段会命中正确页面却排不靠前。
   这正好印证了 Recall@k 和 AnswerHit@k 两个指标**测的不是同一件事**，
   只看一个会得出错误结论。
3. **原来代码里硬编码的 `chunk_size=1000` 是三个里面最差的。**
   这个结论只能靠实验得到——这就是做评测的意义。
4. **混合检索把 Recall@3 从 91.2% 推到 98.2%（57 题里只漏 1 题），MRR 从 0.841 涨到 0.924。**
   收益主要来自 BM25 那一侧：这篇论文里 `BLEU`、`WMT 2014`、`16,384` 这类
   专有名词和数字，向量模型对它们并不敏感（训练语料里见得少），
   但关键词精确匹配一抓一个准。两路互补，所以融合后提升明显。
   **AnswerHit@3 提升 12.3 个百分点**（73.7% → 86.0%）尤其关键——
   这意味着喂给大模型的片段里"真的含有答案原文"的比例大幅提高，
   直接决定了最终生成的准确率。
5. **重排的收益完全取决于模型选得对不对，不能想当然。**
   同一个思路，换模型得到相反结论：`bge-reranker-base` 让 Recall@1 掉 8.8 个百分点，
   换成英文对口的 `ms-marco-MiniLM-L-6-v2` 才涨 3.5 个百分点。
   而且**即使换了对的模型也不是全面胜出**——Recall@3 反而降了 1.7 个百分点。
   这一条是这份评测里最有价值的发现：**任何"常见做法"都要在自家数据上验证一遍。**

---

## 6. 后续路线

- [x] 混合检索：BM25 + 向量（论文里 `BLEU`、`WMT 2014` 这类专有名词，关键词匹配更准）
- [x] 重排（rerank）：召回 20 条后用 cross-encoder 精排——**实测第一个模型是负作用，
      换模型后才涨**，详见 5.3（这条比"重排一定有用"更值得写）
- [x] LangGraph 多跳 Agent：检索 → 判断证据是否充分 → 改写 query → 再检索 → 带出处生成
- [x] 模型可切换：接上 DeepSeek（OpenAI 兼容接口），`LLM_PROVIDER` 一行切换。
      换的动机是一次**控制变量实验**——在**完全相同的证据**下，本地 `qwen2.5:3b`
      **10 次全部**判"证据不够"，DeepSeek 第 1 轮就判对，
      即**"判断力不足"是模型能力问题，不是 prompt 工程问题**
- [x] 答案溯源（基础版）：回答内嵌 `[论文名, p.页码]`，可追溯到片段级 `chunk_id`
- [ ] 🔴 **加 3-5 篇论文（当前唯一瓶颈）**。库里只有一篇论文，答案唾手可得，
      `judge` 每次都在第 1 轮就判"够了"——**多跳与 query 改写从未真正触发过**，
      该 Agent 目前的行为接近 `RAG_v2.py`。主题定为「LLM 奠基主线」：
      `Attention Is All You Need (2017) → BERT (2018) → GPT-3 (2020) → RAG (2020) → ReAct (2022)`
- [ ] 有语料之后再修（已诊断，但**缺语料无法验证**，故暂不动手）：
      - `judge` 会**复读**上一轮的 query（prompt 从未告知它"已经试过什么"）
      - 证据无长度上限，多轮后"证据爆炸"，触发 *lost in the middle*
      - `k = 3n` 的"撒大网"补丁改为固定 `k = 3`，让 query 成为唯一变量
- [ ] 答案溯源（进阶版）：引用可点击跳转
- [ ] 生成质量评测：Faithfulness / Answer Relevancy
- [ ] 多论文支持：跨论文检索与综述
- [ ] 服务化：FastAPI + 流式输出 + Docker 部署

---

## 7. 工作记录

`docs/` 下是逐日的工作记录，写的是**过程和踩坑**，不是结论摘要——
包括方案被推翻、实验做错、以及自己的假设后来站不住的地方。

- [2026-10-07](docs/worklog-2026-10-07.md) — chunk_size 消融、混合检索、重排选型
- [2026-10-08](docs/worklog-2026-10-08.md) — LangGraph 入门、`MAX_HOPS` 刹车、评测可复现性的边界
- [2026-10-09](docs/worklog-2026-10-09.md) — 接上真实检索与 LLM 判断、"判断力"控制变量实验
- [2026-10-10](docs/worklog-2026-10-10.md) — 接入 DeepSeek、provider 工厂、全链路打通

> ⚠️ 两处**已知的诚实性说明**：
> - 5.2 的 `Recall@1` 会跨进程抖动（已加前提说明）；
> - 「多跳 Agent」目前**尚未真正触发多跳**，原因写在 6 的路线里（缺语料）。
