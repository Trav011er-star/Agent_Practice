# Paper-Agent

基于 RAG 的论文阅读助手：把 PDF 论文建成可检索的向量库，支持多轮问答，并且**用一套可复现的评测集量化检索质量**。

当前聚焦单篇论文（*Attention Is All You Need*）打通全链路，后续扩展为多论文文献助手。

---

## 1. 已完成的能力

| 环节 | 实现 | 位置 |
|---|---|---|
| 加载 | PyMuPDFLoader 按页解析 PDF | `src/loader.py` |
| 分块 | RecursiveCharacterTextSplitter，参数可配 | `src/splitter.py` |
| 向量化 | Ollama `nomic-embed-text`（本地） | `src/embedding.py` |
| 存储 | Chroma 本地持久化，按 chunk_size 分集合 | `src/embedding.py` |
| 生成 | Ollama `qwen2.5:3b`（本地） | `src/llm.py` |
| **混合检索** | **自实现 BM25 + 向量检索，RRF 融合** | **`src/retrieval.py`** |
| **重排** | bge-reranker cross-encoder 精排 | `src/retrieval.py` |
| 单轮 RAG | LCEL 管道 `prompt \| llm \| StrOutputParser` | `RAG_v1.py` |
| 多轮 RAG | query rewrite（指代消解）+ 对话历史注入 | `RAG_v2.py` |
| **检索评测** | **57 条标注问题，Recall@k / AnswerHit@k / MRR** | **`eval/`** |
| 建库脚本 | 支持多 chunk_size 批量建库、可重建 | `ingest.py` |

---

## 2. 目录结构

```
paper-Agent/
├── config.py              # 所有可调参数集中在这里
├── ingest.py              # 建库：PDF -> 分块 -> 向量化 -> Chroma
├── RAG_v1.py              # 单轮 RAG
├── RAG_v2.py              # 多轮 RAG（含 query rewrite）
├── requirements.txt
├── src/
│   ├── loader.py          # PDF 解析
│   ├── splitter.py        # 分块
│   ├── embedding.py       # 向量化 + Chroma 读写
│   ├── llm.py             # 大模型
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

---

## 4.5 常见问题排查

所有入口脚本（`ingest.py` / `eval/run_eval.py` / `RAG_v1.py` / `RAG_v2.py`）
启动时都会先调用 `src/health.py` 做自检，把"报错看不懂"的情况翻译成人话。

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
模型：`nomic-embed-text`。**纯检索评测，不涉及生成，结果可完全复现。**

| chunk_size | 片段数 | Recall@1 | Recall@3 | Recall@5 | AnswerHit@1 | AnswerHit@3 | AnswerHit@5 | MRR@10 |
|-----------:|-------:|---------:|---------:|---------:|------------:|------------:|------------:|-------:|
| **500** | 129 | **75.4%** | **91.2%** | **94.7%** | 50.9% | 73.7% | 77.2% | **0.841** |
| 1000 | 52 | 57.9% | 80.7% | 91.2% | 42.1% | 64.9% | 73.7% | 0.706 |
| 2000 | 27 | 54.4% | 84.2% | 93.0% | 45.6% | **75.4%** | **82.5%** | 0.702 |

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
- [ ]答案溯源：回答里附 `[论文名, p.12]` 引用，可点击跳转
- [ ] 生成质量评测：Faithfulness / Answer Relevancy，需要换更强的模型
- [ ] 多论文支持：跨论文检索与综述
- [ ] LangGraph 多跳 Agent：问题拆解 -> 并行检索 -> 汇总 -> 自我检查
- [ ] 服务化：FastAPI + 流式输出 + Docker 部署
