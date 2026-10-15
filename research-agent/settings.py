"""research-agent 的配置。

⚠️ 这个文件【故意】不叫 config.py —— 见下面第二节的注释。
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------
# 一、把 paper-agent 挂到 import 路径上
# ---------------------------------------------------------------
# research-agent 要复用 paper-agent 的检索能力（`agent/engine.py` 里的 retrieve）。
# 但那是另一个目录下的代码，Python 默认看不见它。
#
# 这一行把 paper-agent 的根目录插到搜索路径最前面，之后就能直接：
#     from agent.engine import retrieve
#
# （paper-agent/agent/engine.py 自己也是用同一招把 paper-agent 根目录插进来的，
#   套路一样，只是方向不同。）
PAPER_AGENT_ROOT = Path(__file__).resolve().parent.parent / "paper-agent"
sys.path.insert(0, str(PAPER_AGENT_ROOT))


# ---------------------------------------------------------------
# 二、⚠️ 为什么本文件不叫 config.py
# ---------------------------------------------------------------
# 实测踩过（2026-10-15）：
#
#   paper-agent/agent/engine.py 里有 `import config`，它要的是 paper-agent 的 config。
#   如果 research-agent 里也有个 config.py，且先被 import 过，
#   Python 的 sys.modules 缓存【按名字键，不按路径】——
#   于是 engine.py 的 `import config` 拿到的是 research-agent 这个冒牌货，
#   然后在很深的地方炸出：
#       AttributeError: module 'config' has no attribute 'collection_name_for'
#
#   ImportError 至少告诉你"没找到"，这个连报错都指错地方。
#
# 同理，入口文件也不叫 agent.py —— 那会撞上 paper-agent 的 `agent/` 包。
# 教训：跨项目复用代码时，【通用名】是危险名。settings / main 这类名字更安全。


# ---------------------------------------------------------------
# 三、模型
# ---------------------------------------------------------------
# 直接复用 paper-agent 的密钥文件，不复制密钥内容。
# （同一个密钥写两份 = 将来轮换密钥时漏掉一份）
ENV_FILE = PAPER_AGENT_ROOT / ".env"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


# ---------------------------------------------------------------
# 四、Agent 参数
# ---------------------------------------------------------------
MAX_STEPS = 6      # 最多几轮"模型点单 → 我们执行"（刹车，必须有）
RETRIEVE_K = 3     # query_corpus 每次取回几条片段
