# Agent Practice

面向 **LLM / RAG / AI Agent 应用开发** 的个人学习与项目实践仓库。

本仓库用于记录我从大语言模型基础原理，到 RAG、Agent、LangChain、LangGraph、MCP 等 AI 应用开发技术的学习过程，并通过完整项目逐步将理论知识转化为工程实践能力。

当前重点：

> 🚧 **Paper Agent：基于 LangChain + Ollama + Chroma 构建论文 RAG / Conversational RAG 系统**

---

# 📌 Learning Roadmap

```text
LLM 基础原理
    ↓
Transformer / Token / Embedding
    ↓
LLM API / Local LLM / Ollama
    ↓
Prompt Engineering
    ↓
Tool Calling
    ↓
Agent 基础范式
    ↓
ReAct / Plan-and-Solve / Reflection
    ↓
RAG
    ↓
LangChain / LCEL
    ↓
Conversational RAG
    ↓
LangGraph
    ↓
Agentic RAG
    ↓
MCP
    ↓
Multi-Agent / Agent Engineering
```

仓库根目录中的早期 Python 文件主要用于理解 Agent 与 LLM 应用开发的基础机制，包括：

- LLM API 调用与流式输出
- OpenAI Compatible API
- 本地 / 开源模型调用
- LLM Client 封装
- Tool 定义与动态调用
- Prompt 与上下文管理
- ReAct：Thought → Action → Observation
- Agent Loop
- 基础 Memory
- `.env` 与 API Key 管理

这些代码主要用于理解原理，后续项目将逐步使用 LangChain、LangGraph 等框架进行工程化实现。

---

# 🧠 LLM / Agent 面试知识复习

## 1. Token 与 BPE

### Token 是什么？

Token 是大语言模型处理文本的基本单位。

文本不会直接进入 Transformer，而是：

```text
Text
 ↓
Tokenizer
 ↓
Token
 ↓
Token ID
 ↓
Embedding
 ↓
Vector
```

Token 不一定等于一个完整单词，也可能是：

- 一个字符
- 一个子词
- 一个单词
- 标点符号

### BPE 是什么？

BPE（Byte Pair Encoding）是一类常见的子词分词算法。

基本思想：

> 从基础符号开始，不断合并语料中高频出现的符号组合，最终形成词表。

这样既可以控制词表大小，也能够处理未见过的新词。

---

## 2. Embedding

Embedding 的作用：

> 将离散的 Token / 文本转换成连续的高维向量表示。

例如：

```text
"Transformer uses attention"

        ↓ Embedding

[0.12, -0.34, 0.56, ...]
```

Embedding 模型通过训练，使语义相近的文本在向量空间中距离更近。

因此可以实现：

```text
Query
 ↓
Embedding
 ↓
Query Vector
 ↓
与 Document Vector 比较
 ↓
Semantic Search
```

这也是 Vector RAG 的基础。

---

## 3. Transformer

Transformer 的核心结构：

```text
Token
 ↓
Embedding
 ↓
Position Information
 ↓
Multi-Head Self-Attention
 ↓
Residual + LayerNorm
 ↓
Feed Forward Network
 ↓
Residual + LayerNorm
 ↓
下一层 Transformer
```

### Self-Attention

对于输入向量 `X`：

```text
Q = XWq
K = XWk
V = XWv
```

注意力计算：

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

其中：

- Query：当前 Token 想寻找什么信息
- Key：每个 Token 提供什么匹配信息
- Value：真正被聚合的信息

---

## 4. Multi-Head Attention

单个 Attention Head 只能在一个表示空间中学习关系。

Multi-Head Attention：

```text
Head1
Head2
Head3
...
HeadN
 ↓
Concat
 ↓
Linear Projection
```

不同 Attention Head 可以学习不同类型的 Token 关系。

最终：

```text
MultiHead(Q,K,V)
=
Concat(head1,...,headh)Wo
```

---

## 5. Position Encoding 与 RoPE

Transformer 本身没有 RNN 的顺序结构，因此需要显式加入位置信息。

传统 Position Encoding：

```text
Token Embedding
+
Position Embedding
```

RoPE（Rotary Position Embedding）则通过对 Query / Key 的向量维度进行旋转，引入位置信息。

其特点：

- 不改变向量模长
- 改变向量方向
- Attention 内积能够自然包含相对位置信息

---

## 6. Decoder-only 与 Causal Mask

GPT 类模型通常采用 Decoder-only Transformer。

生成时：

```text
Token1 Token2 Token3 Token4
```

Token4 可以看到：

```text
Token1 Token2 Token3
```

但 Token2 不能看到：

```text
Token3 Token4
```

通过 Causal Mask 实现：

```text
只能关注当前位置左侧的信息
```

从而满足自回归生成：

```text
P(x_t | x_1, x_2, ..., x_(t-1))
```

---

## 7. KV Cache

如果每生成一个 Token 都重新计算之前所有 Token 的 K、V：

```text
Token1
Token1 Token2
Token1 Token2 Token3
...
```

会产生大量重复计算。

KV Cache 会保存历史 Token 已计算好的：

```text
Key
Value
```

生成新 Token 时只需要计算新 Token 的 Q/K/V。

因此 KV Cache：

> 用显存换推理速度。

---

## 8. Context Window

Context Window 表示模型一次推理能够处理的最大 Token 数量。

通常包括：

```text
System Prompt
+
Chat History
+
Retrieved Context
+
Current Query
+
Generated Tokens
```

当上下文过长时：

- 计算成本增加
- KV Cache 占用增加
- 可能超过模型最大上下文长度

这也是 RAG 不直接把整个知识库塞进 Prompt，而是先检索相关内容的重要原因。

---

## 9. Temperature

Temperature 用于调节 Softmax 概率分布：

```text
P(x_i) =
exp(z_i / T)
/
Σ exp(z_j / T)
```

T 较低：

```text
概率分布更集中
→ 输出更稳定
```

T 较高：

```text
概率分布更平滑
→ 输出更多样
```

---

# 🤖 Agent 基础

普通 LLM：

```text
User
 ↓
LLM
 ↓
Answer
```

Agent：

```text
User
 ↓
LLM
 ↓
Reasoning / Planning
 ↓
Tool
 ↓
Observation
 ↓
LLM
 ↓
...
 ↓
Answer
```

Agent 的核心不是“模型更大”，而是：

> LLM 可以根据任务状态决定下一步行动，并利用外部工具与环境交互。

---

## ReAct

ReAct：

```text
Reasoning + Acting
```

典型循环：

```text
Thought
 ↓
Action
 ↓
Observation
 ↓
Thought
 ↓
Action
 ↓
...
 ↓
Final Answer
```

---

## Plan-and-Solve

与 ReAct 边执行边思考不同：

```text
问题
 ↓
先制定完整计划
 ↓
按计划逐步执行
```

更适合结构明确的复杂任务。

---

## Reflection

Reflection 强调：

```text
生成结果
 ↓
检查结果
 ↓
发现问题
 ↓
修改
 ↓
重新执行
```

适合：

- Coding
- Code Review
- 写作
- 推理检查
- 自我纠错

---

# 🔌 Tool Calling 与 MCP

## Tool Calling

LLM 本身不会直接执行 Python 函数。

它产生：

```text
Tool Name
+
Arguments
```

本地程序负责真正执行函数，并把结果重新返回给模型。

---

## MCP

MCP：

```text
Model Context Protocol
```

用于标准化：

```text
Agent
 ↓
外部工具 / 数据 / 系统
```

可以理解为：

> Agent 连接外部能力的一套标准协议。

例如：

```text
Agent
 ↓
MCP Client
 ↓
MCP Server
 ├── File System
 ├── GitHub
 ├── Database
 └── Search
```

---

# 📚 RAG

RAG：

```text
Retrieval-Augmented Generation
```

核心思想：

> 先检索相关知识，再让 LLM 基于检索结果生成答案。

基本流程：

```text
Documents
 ↓
Chunk
 ↓
Embedding
 ↓
Vector Store
```

查询：

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Relevant Chunks
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

RAG 不一定需要访问互联网。

知识源可以是：

```text
PDF
Markdown
Database
Enterprise Documents
Vector Database
Web Search
```

---

# 🚀 Projects

## 01. Paper Agent 🚧 In Progress

目录：

```text
paper-Agent/
```

目标：

> 从零实现一个面向论文阅读的 Local RAG / Conversational RAG 系统，并逐步升级为 Research Agent。

### 技术栈

```text
Python
LangChain
LCEL
Ollama
Chroma
PyMuPDF
nomic-embed-text
```

---

# 📖 Paper Agent 学习记录

## Stage 1：PDF Loading

使用：

```python
PyMuPDFLoader
```

将 PDF 转换成 LangChain：

```python
List[Document]
```

每个 Document 主要包含：

```text
page_content
metadata
```

例如：

```text
Document
├── page_content
└── metadata
    ├── page
    └── source
```

Metadata 用于保留页码、来源等信息，为后续 Citation 提供基础。

---

## Stage 2：Text Splitting

使用：

```python
RecursiveCharacterTextSplitter
```

将长文档切分成多个 Chunk：

```text
PDF
 ↓
Document
 ↓
Chunk1
Chunk2
Chunk3
...
```

主要参数：

```text
chunk_size
chunk_overlap
```

Overlap 可以降低文本边界切断语义的问题。

---

## Stage 3：Embedding

当前使用本地 Ollama：

```text
nomic-embed-text
```

流程：

```text
Chunk
 ↓
Embedding Model
 ↓
High-dimensional Vector
```

每一个 Chunk 对应一个语义向量。

Query 使用同一个 Embedding Model：

```text
Query
 ↓
Query Vector
```

才能在同一个语义空间中进行距离比较。

---

## Stage 4：Chroma Vector Store

使用：

```python
Chroma
```

存储：

```text
ID
Embedding
Document
Metadata
```

区分两个过程：

### Offline Indexing

```text
PDF
 ↓
Chunk
 ↓
Embedding
 ↓
Chroma
```

只在建立 / 更新知识库时执行。

### Online Retrieval

```text
Query
 ↓
Query Embedding
 ↓
Chroma Search
 ↓
Top-K Documents
```

避免每次查询都重新进行文档 Embedding。

---

## Stage 5：Retriever

最初直接使用：

```python
vector_store.similarity_search()
```

后升级为：

```python
retriever = vector_store.as_retriever()
```

Retriever 是 LangChain 对“检索行为”的统一抽象。

Retriever 实现了 Runnable 接口，因此可以：

```python
retriever.invoke(query)
```

并直接加入 LCEL 工作流。

---

## Stage 6：Runnable 与 LCEL

LangChain 中：

```text
Runnable
```

是可执行组件的统一抽象。

常见 Runnable：

```text
Retriever
PromptTemplate
ChatModel
RunnableLambda
RunnablePassthrough
Chain
```

因此可以：

```python
chain = prompt | llm
```

`|` 表示：

```text
前一个 Runnable 输出
        ↓
后一个 Runnable 输入
```

### RunnableLambda

将普通 Python 函数包装成 Runnable：

```python
RunnableLambda(format_docs)
```

从而加入 LCEL Pipeline。

### RunnablePassthrough

输入什么，就原样返回什么：

```text
input
 ↓
RunnablePassthrough
 ↓
input
```

常用于工作流分支中保留原始输入。

---

## Stage 7：RAG Chain

当前实现的基本 LCEL：

```text
                      Query
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
          Retriever         Passthrough
              ↓                   ↓
       List[Document]            Query
              ↓
        Format Documents
              ↓
           Context
              │                   │
              └─────────┬─────────┘
                        ↓
                 PromptTemplate
                        ↓
                       LLM
                        ↓
                StrOutputParser
                        ↓
                     Answer
```

---

## Stage 8：Chat History

加入：

```python
HumanMessage
AIMessage
MessagesPlaceholder
```

聊天记录：

```python
[
    HumanMessage(...),
    AIMessage(...),
    HumanMessage(...),
    AIMessage(...)
]
```

Prompt：

```python
MessagesPlaceholder(
    variable_name="chat_history"
)
```

保留真实的：

```text
Human
AI
Human
AI
```

消息角色关系。

实现：

```text
Chat History
     ↓
Final LLM
```

从单轮 RAG 升级为多轮 Conversational RAG。

---

## Stage 9：History-aware Retrieval

仅仅让最终 LLM 看到聊天历史还不够。

例如：

```text
Human:
What is multi-head attention?

Human:
Why is it useful?
```

如果 Retriever 直接搜索：

```text
Why is it useful?
```

语义信息不足。

因此加入 Query Rewrite：

```text
Chat History
+
Current Query
 ↓
LLM Rewrite
 ↓
Standalone Query
```

例如：

```text
Why is it useful?

↓

What are the benefits of using multi-head attention?
```

然后：

```text
Rewritten Query
 ↓
Retriever
 ↓
Relevant Documents
```

最终：

```text
Chat History
        ↓
   Query Rewrite
        ↓
Standalone Query
        ↓
    Retriever
        ↓
     Context
        │
        ├──────────────┐
        │              │
Original Query    Chat History
        │              │
        └──────┬───────┘
               ↓
             Prompt
               ↓
              LLM
               ↓
             Answer
```

当前项目已经完成这一阶段。

---

# 🔨 Paper Agent 下一步

计划继续加入：

```text
Source Citation
        ↓
Similarity Score / Threshold
        ↓
MMR Retrieval
        ↓
Reranker
        ↓
Multiple PDFs
        ↓
Document Management
        ↓
LangGraph
        ↓
Agentic RAG
        ↓
Web / arXiv Search Tool
        ↓
Research Agent
```

最终目标：

```text
User Research Question
        ↓
Query Analysis
        ↓
Local Paper RAG
        ↓
External Paper Search
        ↓
Evidence Retrieval
        ↓
Answer / Summary
        ↓
Citation
        ↓
Research Report
```

---

# 🗺️ Project Roadmap

| Project | Core Technologies | Status |
|---|---|---|
| **Paper Agent** | LangChain, RAG, Chroma, Ollama, Conversational RAG | 🚧 In Progress |
| **Research Agent** | LangGraph, Tool Calling, Planning, Web / arXiv Search | 📋 Planned |
| **MCP Agent** | MCP Client / Server, Files, GitHub, Database Tools | 📋 Planned |
| **Multi-Agent Workflow** | LangGraph, Multi-Agent, Reflection, Human-in-the-loop | 📋 Planned |

---

# 🧩 Current Tech Stack

```text
Language
└── Python

LLM
├── Ollama
├── OpenAI Compatible API
└── Local Open-source Models

LLM Framework
├── LangChain
└── LCEL

RAG
├── PyMuPDF
├── RecursiveCharacterTextSplitter
├── nomic-embed-text
├── Chroma
└── Retriever

Agent
├── ReAct
├── Tool Calling
├── Memory
└── Query Rewrite

Next
├── LangGraph
├── MCP
├── Agentic RAG
└── Multi-Agent
```

---

# 🎯 Repository Goal

这个仓库并不是简单收集各种 AI Demo。

主要目标是：

> 从底层机制开始理解 LLM 与 Agent，再逐步使用框架进行工程化实现，并通过完整项目形成可以解释、可以扩展、可以复现的 AI Agent 开发能力。

学习方式：

```text
理解原理
 ↓
手动实现
 ↓
框架重构
 ↓
发现问题
 ↓
逐步优化
 ↓
完整项目
```

后续会持续更新 Paper Agent，并逐步推进 LangGraph、Agentic RAG、MCP 与 Multi-Agent 项目。
