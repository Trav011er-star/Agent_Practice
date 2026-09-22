# Agent Practice

这是我的 **LLM / RAG / AI Agent 学习与项目实践仓库**。

仓库主要分为两部分：

1. **基础知识**：记录已经学习过的核心概念，用于日常复习和面试准备。
2. **项目实践**：记录每个项目从设计、实现到优化的完整过程。

后续每学习一个新的概念，例如 **RAG、MCP、LangGraph、Reranker、Multi-Agent**，都会继续补充到“基础知识”中；每完成一个新的项目或功能，则继续补充到对应项目章节。

---

# 目录

## 1. 基础知识

### 1.1 LLM 基础
- [Token](#111-token)
- [BPE](#112-bpe)
- [Embedding](#113-embedding)
- [Transformer](#114-transformer)
- [Self-Attention](#115-self-attention)
- [Multi-Head Attention](#116-multi-head-attention)
- [Position Encoding 与 RoPE](#117-position-encoding-与-rope)
- [Decoder-only 与 Causal Mask](#118-decoder-only-与-causal-mask)
- [KV Cache](#119-kv-cache)
- [Context Window](#1110-context-window)
- [Temperature](#1111-temperature)

### 1.2 LLM 应用基础
- [Prompt Engineering](#121-prompt-engineering)
- [LLM API 与 OpenAI Compatible API](#122-llm-api-与-openai-compatible-api)
- [Ollama 与本地模型](#123-ollama-与本地模型)

### 1.3 Agent 基础
- [什么是 Agent](#131-什么是-agent)
- [Tool Calling / Function Calling](#132-tool-calling--function-calling)
- [ReAct](#133-react)
- [Plan-and-Solve](#134-plan-and-solve)
- [Reflection](#135-reflection)
- [Memory](#136-memory)

### 1.4 RAG 基础
- [什么是 RAG](#141-什么是-rag)
- [Document 与 Chunk](#142-document-与-chunk)
- [Embedding 与语义检索](#143-embedding-与语义检索)
- [Vector Database](#144-vector-database)
- [Retriever](#145-retriever)
- [Conversational RAG](#146-conversational-rag)
- [History-aware Retrieval](#147-history-aware-retrieval)

### 1.5 LangChain / Agent 工程
- [LangChain](#151-langchain)
- [LCEL](#152-lcel)
- [Runnable](#153-runnable)
- [RunnableLambda](#154-runnablelambda)
- [RunnablePassthrough](#155-runnablepassthrough)

### 1.6 MCP
- [MCP 是什么](#161-mcp-是什么)
- [MCP Client / Server](#162-mcp-client--server)
- [MCP 与 Tool Calling 的区别](#163-mcp-与-tool-calling-的区别)

---

## 2. Paper Agent

- [2.1 项目目标](#21-项目目标)
- [2.2 技术栈](#22-技术栈)
- [2.3 当前系统架构](#23-当前系统架构)
- [2.4 Stage 1：PDF Loading](#24-stage-1pdf-loading)
- [2.5 Stage 2：Text Splitting](#25-stage-2text-splitting)
- [2.6 Stage 3：Embedding](#26-stage-3embedding)
- [2.7 Stage 4：Chroma Vector Store](#27-stage-4chroma-vector-store)
- [2.8 Stage 5：Retriever](#28-stage-5retriever)
- [2.9 Stage 6：LCEL RAG Chain](#29-stage-6lcel-rag-chain)
- [2.10 Stage 7：Chat History](#210-stage-7chat-history)
- [2.11 Stage 8：History-aware Retrieval](#211-stage-8history-aware-retrieval)
- [2.12 下一步计划](#212-下一步计划)

---

## 3. 后续项目

- [3.1 Research Agent](#31-research-agent)
- [3.2 MCP Agent](#32-mcp-agent)
- [3.3 Multi-Agent Workflow](#33-multi-agent-workflow)

---

# 1. 基础知识

## 1.1 LLM 基础

### 1.1.1 Token

Token 是大语言模型处理文本的基本单位。

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

Token 不一定是完整单词，也可能是字符、子词或标点。

---

### 1.1.2 BPE

BPE（Byte Pair Encoding）是一类常见的子词分词算法。

核心思想：

> 从基础符号开始，不断合并语料中高频出现的符号组合，逐步形成词表。

这样既可以控制词表规模，也能够处理未见过的新词。

---

### 1.1.3 Embedding

Embedding 将离散的 Token 或文本映射为连续的高维向量。

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

语义相近的文本会被训练到向量空间中较近的位置，因此可以用于语义检索。

---

### 1.1.4 Transformer

Transformer 的核心结构可以概括为：

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
```

与 RNN 不同，Transformer 不依赖递归结构，而是通过 Attention 建模 Token 之间的关系。

---

### 1.1.5 Self-Attention

对于输入矩阵 `X`：

```text
Q = XWq
K = XWk
V = XWv
```

Attention：

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

直观理解：

- Query：当前 Token 想寻找什么信息
- Key：每个 Token 能提供什么匹配信息
- Value：真正被聚合的信息

---

### 1.1.6 Multi-Head Attention

Multi-Head Attention 使用多个 Attention Head 并行学习不同表示空间中的关系。

```text
Head1
Head2
...
HeadN
 ↓
Concat
 ↓
Linear Projection
```

公式：

```text
MultiHead(Q,K,V)
=
Concat(head1,...,headh)Wo
```

多个 Head 可以关注不同位置、不同语义关系和不同特征子空间。

---

### 1.1.7 Position Encoding 与 RoPE

Transformer 本身没有顺序结构，因此需要加入位置信息。

传统方法：

```text
Token Embedding
+
Position Encoding
```

RoPE（Rotary Position Embedding）通过旋转 Query / Key 的向量表示引入位置信息。

特点：

- 不改变向量模长
- 改变向量方向
- Attention 内积可以自然包含相对位置信息

---

### 1.1.8 Decoder-only 与 Causal Mask

GPT 类模型通常采用 Decoder-only Transformer。

Causal Mask 保证：

```text
当前位置只能看到当前及之前的 Token
不能看到未来 Token
```

因此能够实现自回归生成：

```text
P(x_t | x_1, x_2, ..., x_(t-1))
```

---

### 1.1.9 KV Cache

生成新 Token 时，历史 Token 的 Key 和 Value 不需要重复计算。

KV Cache 保存历史：

```text
Key
Value
```

新 Token 生成时只计算新增部分。

本质上：

> 使用显存换取推理速度。

---

### 1.1.10 Context Window

Context Window 是模型一次能够处理的最大 Token 范围。

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

上下文越长，推理成本和 KV Cache 占用通常越高。

---

### 1.1.11 Temperature

Temperature 用于调整模型采样时的概率分布。

```text
P(x_i)
=
exp(z_i / T)
/
Σ exp(z_j / T)
```

- `T` 较低：概率分布更集中，输出更稳定
- `T` 较高：概率分布更平滑，输出更多样

---

## 1.2 LLM 应用基础

### 1.2.1 Prompt Engineering

Prompt Engineering 是通过设计输入指令，引导 LLM 按预期方式完成任务。

常见内容包括：

- System Prompt
- 任务说明
- 输出格式约束
- Few-shot 示例
- Context 注入
- 角色与边界约束

Prompt Engineering 是 LLM 应用开发中的一个环节，但并不等于完整的 LLM 应用工程。

---

### 1.2.2 LLM API 与 OpenAI Compatible API

LLM 应用通常通过 API 与模型服务通信：

```text
Application
 ↓
SDK / HTTP
 ↓
LLM API
 ↓
Model
```

OpenAI Compatible API 允许不同模型服务使用相似的接口格式，从而降低应用层的切换成本。

---

### 1.2.3 Ollama 与本地模型

Ollama 用于在本地运行和管理开源模型。

```text
Python Application
 ↓
Ollama Client
 ↓
Ollama Local Service
 ↓
Local Model
```

Python 虚拟环境中安装的是客户端依赖，而 Ollama 本身作为独立服务运行。

---

## 1.3 Agent 基础

### 1.3.1 什么是 Agent

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

Agent 的核心能力是：

> 根据当前任务状态决定下一步行动，并利用外部工具与环境完成任务。

---

### 1.3.2 Tool Calling / Function Calling

LLM 通常不会直接执行 Python 函数，而是生成：

```text
Tool Name
+
Arguments
```

应用程序负责：

```text
LLM 决定调用工具
 ↓
程序执行工具
 ↓
获得结果
 ↓
把结果返回 LLM
```

---

### 1.3.3 ReAct

ReAct = Reasoning + Acting。

典型流程：

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

特点是推理和行动交替进行。

---

### 1.3.4 Plan-and-Solve

Plan-and-Solve 更强调先规划，再执行。

```text
Problem
 ↓
Plan
 ↓
Step 1
 ↓
Step 2
 ↓
...
 ↓
Answer
```

适用于结构较明确、需要多步骤执行的任务。

---

### 1.3.5 Reflection

Reflection 强调对已有结果进行检查和修正。

```text
Generate
 ↓
Evaluate
 ↓
Find Problems
 ↓
Revise
 ↓
Generate Again
```

常用于代码生成、代码审查、写作和复杂推理。

---

### 1.3.6 Memory

Agent / Chat 应用中的 Memory 用于保存过去交互产生的信息。

短期会话历史通常可以表示为：

```text
HumanMessage
AIMessage
HumanMessage
AIMessage
...
```

后续还可以扩展为摘要 Memory、向量 Memory、长期 Memory 等。

---

## 1.4 RAG 基础

### 1.4.1 什么是 RAG

RAG：

> Retrieval-Augmented Generation

核心思想：

```text
先检索相关知识
 ↓
把知识加入 Prompt
 ↓
LLM 基于这些知识回答
```

RAG 不要求知识一定来自互联网，本地 PDF、数据库、企业文档都可以作为知识源。

---

### 1.4.2 Document 与 Chunk

长文档通常不会整体直接用于检索，而是先切分成 Chunk。

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

Chunk 可以提升检索粒度，并降低上下文长度。

---

### 1.4.3 Embedding 与语义检索

```text
Document Chunk
 ↓
Embedding
 ↓
Document Vector
```

查询时：

```text
Query
 ↓
Embedding
 ↓
Query Vector
 ↓
Similarity Search
```

Query 和 Document 应使用兼容的 Embedding 模型映射到同一向量空间。

---

### 1.4.4 Vector Database

向量数据库用于存储：

```text
ID
Vector
Document
Metadata
```

当前项目使用：

```text
Chroma
```

---

### 1.4.5 Retriever

Retriever 是对“检索行为”的统一抽象。

输入：

```text
Query
```

输出：

```text
List[Document]
```

它不一定只能使用向量检索，也可以基于 BM25、SQL、Web Search 等方式实现。

---

### 1.4.6 Conversational RAG

普通 RAG：

```text
Current Question
 ↓
Retriever
 ↓
LLM
```

Conversational RAG 还需要利用历史对话：

```text
Chat History
+
Current Question
+
Retrieved Context
 ↓
LLM
```

---

### 1.4.7 History-aware Retrieval

仅让最终 LLM 看到历史还不够。

例如：

```text
Q1: What is multi-head attention?
Q2: Why is it useful?
```

如果 Retriever 只搜索：

```text
Why is it useful?
```

信息不足。

因此加入 Query Rewrite：

```text
Chat History
+
Current Question
 ↓
LLM Rewrite
 ↓
Standalone Question
 ↓
Retriever
```

例如：

```text
Why is it useful?
 ↓
What are the benefits of using multi-head attention?
```

---

## 1.5 LangChain / Agent 工程

### 1.5.1 LangChain

LangChain 是用于构建 LLM 应用的开发框架，提供：

- Prompt
- Model
- Retriever
- Document
- Output Parser
- Runnable
- Tool
- Agent 等抽象

---

### 1.5.2 LCEL

LCEL：

> LangChain Expression Language

通过 `|` 将多个组件组合成工作流：

```python
chain = prompt | llm | parser
```

含义：

```text
上一个组件的输出
 ↓
下一个组件的输入
```

---

### 1.5.3 Runnable

Runnable 是 LangChain 对“可执行组件”的统一抽象。

Runnable 可以：

```python
component.invoke(input)
```

Prompt、LLM、Retriever 等组件都可以作为 Runnable 接入工作流。

---

### 1.5.4 RunnableLambda

RunnableLambda 将普通 Python 函数包装成 Runnable：

```python
RunnableLambda(function)
```

从而使普通函数能够加入 LCEL 工作流。

---

### 1.5.5 RunnablePassthrough

RunnablePassthrough：

```text
输入什么
 ↓
原样输出什么
```

通常用于工作流分支中保留原始输入。

---

## 1.6 MCP

### 1.6.1 MCP 是什么

MCP：

> Model Context Protocol

用于标准化 AI 应用与外部工具、数据源和系统之间的连接方式。

```text
Agent
 ↓
MCP Client
 ↓
MCP Server
 ↓
External Capability
```

---

### 1.6.2 MCP Client / Server

MCP Client 通常存在于 AI 应用一侧。

MCP Server 对外暴露可用能力，例如：

```text
File System
Database
GitHub
Search
Other Services
```

---

### 1.6.3 MCP 与 Tool Calling 的区别

Tool Calling 主要解决：

> 模型如何决定调用哪个工具，以及传递什么参数。

MCP 主要解决：

> 工具和外部资源如何以标准协议提供给 AI 应用。

两者可以配合使用。

---

# 2. Paper Agent

## 2.1 项目目标

目录：

```text
paper-Agent/
```

当前目标：

> 从零构建一个面向论文阅读的 Local RAG / Conversational RAG 系统，并在后续逐步升级为 Research Agent。

---

## 2.2 技术栈

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

## 2.3 当前系统架构

```text
PDF
 ↓
PyMuPDFLoader
 ↓
Documents
 ↓
Text Splitter
 ↓
Chunks
 ↓
Embedding
 ↓
Chroma Vector Store
```

查询流程：

```text
User Question
 ↓
Chat History + Query Rewrite
 ↓
Standalone Retrieval Query
 ↓
Retriever
 ↓
Relevant Chunks
 ↓
Context
 ↓
Prompt + Original Question + Chat History
 ↓
LLM
 ↓
Answer
```

---

## 2.4 Stage 1：PDF Loading

使用：

```python
PyMuPDFLoader
```

读取 PDF 后得到：

```python
List[Document]
```

每个 `Document` 主要包含：

```text
page_content
metadata
```

Metadata 保存页码和来源等信息，为后续 Source Citation 提供基础。

---

## 2.5 Stage 2：Text Splitting

使用：

```python
RecursiveCharacterTextSplitter
```

当前核心参数：

```text
chunk_size
chunk_overlap
```

`chunk_overlap` 用于降低文本刚好在 Chunk 边界被切断时造成的语义损失。

---

## 2.6 Stage 3：Embedding

使用 Ollama：

```text
nomic-embed-text
```

```text
Chunk
 ↓
Embedding Model
 ↓
Vector
```

Query 也使用同一 Embedding Model 转换为向量，然后与文档向量进行语义相似度比较。

---

## 2.7 Stage 4：Chroma Vector Store

当前向量数据库：

```text
Chroma
```

将流程分为两部分：

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

只在建立或更新知识库时进行。

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

查询时无需重新计算所有文档向量。

---

## 2.8 Stage 5：Retriever

将：

```python
vector_store.similarity_search(...)
```

进一步封装为：

```python
retriever = vector_store.as_retriever(...)
```

Retriever 接收 Query，并返回：

```python
List[Document]
```

同时可以作为 Runnable 直接加入 LCEL 工作流。

---

## 2.9 Stage 6：LCEL RAG Chain

通过 LCEL 将 RAG 各组件串联：

```text
Query
 ├───────────────┐
 ↓               ↓
Retriever      Question
 ↓
Documents
 ↓
Format Docs
 ↓
Context
 └───────┬───────┘
         ↓
       Prompt
         ↓
        LLM
         ↓
StrOutputParser
         ↓
       Answer
```

---

## 2.10 Stage 7：Chat History

加入：

```python
HumanMessage
AIMessage
MessagesPlaceholder
```

保存历史：

```text
Human
AI
Human
AI
...
```

Prompt 中通过：

```python
MessagesPlaceholder(variable_name="chat_history")
```

将真实角色消息插入对话上下文。

---

## 2.11 Stage 8：History-aware Retrieval

为了解决追问中的代词和上下文依赖，增加 Query Rewrite。

测试：

```text
Q1:
What is multi-head attention?

Q2:
Why is it useful?
```

系统成功改写为：

```text
What are the benefits of using multi-head attention?
```

进一步测试：

```text
Q3:
How are its outputs combined?
```

能够改写成与 Multi-Head Attention 明确相关的独立问题。

当前完整流程：

```text
Original Query
      │
      ├────────────→ Final Question
      │
      ↓
Chat History
      ↓
Query Rewrite
      ↓
Standalone Query
      ↓
Retriever
      ↓
Relevant Context
      │
      └────────────┐
                   ↓
            Final Prompt
          ├ Context
          ├ Chat History
          └ Original Query
                   ↓
                  LLM
                   ↓
                 Answer
```

至此已经完成：

> **Conversational RAG + History-aware Retrieval**

---

## 2.12 下一步计划

接下来继续完善：

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
Web / arXiv Search
 ↓
Research Agent
```

随着项目继续开发，本章节会持续记录每一次新增功能、设计原因和实现过程。

---

# 3. 后续项目

## 3.1 Research Agent

计划在 Paper Agent 基础上进一步加入：

```text
LangGraph
Tool Calling
Planning
Web Search
arXiv Search
Evidence Collection
Citation
Research Report
```

目标：

```text
Research Question
 ↓
Plan
 ↓
Search
 ↓
Retrieve
 ↓
Read
 ↓
Synthesize
 ↓
Citation
 ↓
Report
```

---

## 3.2 MCP Agent

计划通过 MCP 将 Agent 与不同外部能力连接：

```text
Agent
 ↓
MCP Client
 ↓
MCP Servers
 ├── Files
 ├── GitHub
 ├── Database
 └── Search
```

重点学习 MCP 的实际工程使用方式，而不仅停留在协议概念。

---

## 3.3 Multi-Agent Workflow

后续计划进一步学习：

```text
Multi-Agent
LangGraph
Agent Collaboration
Reflection
Human-in-the-loop
State Management
```

通过多个具有不同职责的 Agent 协同完成复杂任务。

---

# 学习方式

整个仓库采用：

```text
理解概念
 ↓
手动实现
 ↓
观察运行结果
 ↓
发现问题
 ↓
引入新技术解决问题
 ↓
记录项目演进
```

基础知识部分用于持续积累 **面试知识体系**，项目部分用于记录 **真实工程实践过程**。

后续学习到新的概念或完成新的项目功能时，会继续更新本 README。
