"""
项目统一配置。

设计原则：所有"可以调的参数"都集中在这一个文件里。
以后想试 chunk_size = 500，只改这里一行，不用去翻 6 个源文件。

注意：路径一律用 PROJECT_ROOT 拼绝对路径，这样不管你在哪个目录下敲命令都能跑。
（原来的代码用的是 "vectorstore/chroma_db" 这种相对路径，换个目录运行就找不到文件了。）
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "papers"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULT_DIR = EVAL_DIR / "results"

# ---------- 环境变量 ----------
load_dotenv(PROJECT_ROOT / ".env")

# ---------- 数据 ----------
DEFAULT_PDF = DATA_DIR / "Attention Is All You Need.pdf"

# ---------- 分块（chunking）----------
# chunk_size 越大，单个片段装的信息越多但越"杂"；越小越精准但可能丢上下文。
#
# 这个值不是拍脑袋定的，是靠 eval/run_eval.py 跑实验比出来的（57 题，见 README 5.2）：
#   500  -> Recall@1 75.4%  Recall@3 91.2%  MRR 0.841   <- 最优
#   1000 -> Recall@1 57.9%  Recall@3 80.7%  MRR 0.706   <- 原硬编码值，三个里最差
#   2000 -> Recall@1 54.4%  Recall@3 84.2%  MRR 0.702
# 所以默认取 500。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

# ---------- 检索 ----------
# 最终喂给大模型几个片段
RETRIEVE_K = 3
# 评测时先多捞一些，才能算 Recall@5 / MRR@10
EVAL_RETRIEVE_K = 10

# ---------- 混合检索 ----------
# 向量检索和 BM25 各召回多少条用于 RRF 融合。
# 这个值要明显大于最终要的 k（重排是"从多里挑精"），但不能太大否则重排变慢。
HYBRID_CANDIDATE_K = 20

# RRF 公式里的平滑常数，原论文（Cormack et al. 2009）取的 60，一般不用调
RRF_K = 60

# ---------- 重排 ----------
# 首次运行会从 HuggingFace 下载（约 90MB）。
#
# 这里试过两个模型，实测（chunk_size=500，57 题，见 eval/rerank_compare.py）：
#
#   BAAI/bge-reranker-base           R@1 78.9%  MRR 0.860   <- 比不重排还差
#   cross-encoder/ms-marco-MiniLM-L-6-v2  R@1 91.2%  MRR 0.939   <- 用它
#   （不重排的 hybrid 基线）          R@1 87.7%  MRR 0.924
#
# bge-reranker-base 虽然在国内更出名，但训练语料以中文为主，
# 放到英文论文上是**负作用**——不但没提升，还把指标拉低了。
# ms-marco 是在英文 passage ranking 上训练的，方向才对得上。
#
# 结论：重排模型必须按语言/领域选，还要实测，不能看到"cross-encoder"就以为一定涨。
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# huggingface_hub 从哪个站点下载模型。
# 踩坑记录：国内直连 huggingface.co 会一直挂到超时（实测 12s 无响应），
# 换成镜像站 0.6s 就返回了。
# 注意：huggingface_hub 在 **import 时** 就把 HF_ENDPOINT 读成了常量，
# 所以必须在导入它之前设好——这里放在 config 顶层，保证任何入口脚本
# （都会先 import config）都能生效。
HF_ENDPOINT = "https://hf-mirror.com"
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# ---------- 模型 ----------
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

# ---------- Ollama 服务地址 ----------
# 这里必须显式写死，不能依赖环境变量 OLLAMA_HOST。
#
# 踩坑记录：如果系统里设了 OLLAMA_HOST=0.0.0.0:11434（常见做法，用来把 Ollama
# 暴露到局域网），ollama 的 Python 客户端会照着它去连 http://0.0.0.0:11434。
# 但 0.0.0.0 是"服务端监听地址"，不是"客户端连接地址"——在 Windows 上连接它会
# 直接失败：
#     ConnectError [WinError 10049] 在其上下文中，该请求的地址无效。
# 而客户端还会把这个异常吞掉，统一报 "Failed to connect to Ollama"，
# 让人完全排查不到原因。
#
# 显式传入 base_url 后，本项目就不受这个环境变量影响了。
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# ---------- 向量库命名 ----------
# 每种 chunk_size 用一个独立的 collection，互不覆盖，这样才能做对比实验。
COLLECTION_PREFIX = "attention_paper"


def collection_name_for(chunk_size: int) -> str:
    """按 chunk_size 生成集合名，例如 attention_paper_1000"""
    return f"{COLLECTION_PREFIX}_{chunk_size}"


# 默认集合名（对应默认的 CHUNK_SIZE）
COLLECTION_NAME = collection_name_for(CHUNK_SIZE)

# ---------- 云端模型（DeepSeek）----------
# 密钥从 .env 读，绝不写进代码。
# .env 已经在 .gitignore 里，不会被提交。
# 前面依旧通过 load_dotenv 把 API_KEY 读进环境变量
# 这里的环境变量是指 python.exe 进程中的，他会复制系统的环境/用户变量，并优先使用系统里的
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ---------- 用哪家（全局开关）----------
# "ollama"   → 本地 qwen2.5:3b
# "deepseek" → 云端 API
LLM_PROVIDER = "deepseek"
