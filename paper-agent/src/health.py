"""启动前自检：确认 Ollama 服务可用、需要的模型已下载。

为什么需要这个模块？
--------------------------------------------------
如果 Ollama 没启动，`ollama` 的 Python 客户端**不会**报"连接失败"。
在开着系统代理的机器上（比如 Clash 监听 127.0.0.1:7897），
发往 localhost 的请求会被代理接管，代理连不上后端就返回一个 502，
最终报错信息长这样：

    ollama._types.ResponseError:  (status code: 502)

完全看不出"是 Ollama 没开"。这个模块把它翻译成人话，
在程序启动第一步就拦住，而不是等跑到一半才炸。

这里用 socket 直接探端口（不经过 HTTP，天然绕过代理），
用 urllib 时也显式禁用代理，保证探测结果反映的是真实情况。
"""

import json
import os
import socket
import urllib.parse
import urllib.request

import config

# 从 config 的统一地址解析出 host / port，保证探测和实际调用的是同一个地址
_parsed = urllib.parse.urlparse(config.OLLAMA_BASE_URL)
OLLAMA_HOST = _parsed.hostname or "127.0.0.1"
OLLAMA_PORT = _parsed.port or 11434


class OllamaNotReady(RuntimeError):
    """Ollama 不可用，附带给用户看的处理建议。"""


def _port_open(host=None, port=None, timeout=2.0):
    """直接建 TCP 连接探测端口，不经过 HTTP 代理。

    注意：host/port 默认值在函数**调用时**才从模块变量取，
    不能写成 `def _port_open(host=OLLAMA_HOST, port=OLLAMA_PORT)`——
    那样默认值会在定义时被固定住，改了模块变量也不生效。
    """
    host = host or OLLAMA_HOST
    port = port or OLLAMA_PORT
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def list_local_models():
    """返回本地已下载的模型名列表，例如 ['qwen2.5:3b', 'nomic-embed-text:latest']。"""
    # ProxyHandler({}) 表示"一个代理都不用"，否则会被系统代理劫持
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    with opener.open(url, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", [])]


_SERVICE_DOWN_HINT = """\
Ollama 服务没有启动（{host}:{port} 连不上）。

请任选一种方式启动：
  1. 打开 Ollama 桌面版（推荐，它会自动带上你的模型目录配置）
  2. 在另一个终端手动启动：
       默认模型目录：      ollama serve
       模型在别的盘：      PowerShell:  $env:OLLAMA_MODELS = "E:\\Ollama\\models"; ollama serve
                          CMD:         set OLLAMA_MODELS=E:\\Ollama\\models && ollama serve

排查提示：
  * 如果你的报错是 "status code: 502" 而不是"连接失败"，那是系统代理
    （HTTP_PROXY / Windows 代理设置）把发往 localhost 的请求也转发走了，
    代理连不上后端就回 502。根因同样是 Ollama 没启动，不要被这个错误码带偏。
  * 如果启动后报"找不到模型"，说明 OLLAMA_MODELS 指向的目录不对，
    用上面的第 2 种方式启动即可。"""

_ENV_HOST_HINT = """\
检测到环境变量 OLLAMA_HOST={value}，它和本项目配置的地址不一致。

OLLAMA_HOST 是**服务端监听地址**（用 0.0.0.0 表示"监听所有网卡"），
不是**客户端连接地址**。ollama 的 Python 客户端会读这个变量去连接，
于是连到 http://0.0.0.0:11434 —— 在 Windows 上这会直接失败：
    ConnectError [WinError 10049] 在其上下文中，该请求的地址无效。

本项目已经在 config.OLLAMA_BASE_URL 里显式写死了连接地址，所以不受影响。
但其它用到 ollama 客户端的工具仍会踩这个坑。如果不需要把 Ollama 暴露到
局域网，建议把这个用户环境变量改掉或删掉：

    # 查看
    reg query "HKCU\\Environment" /v OLLAMA_HOST
    # 改成只监听本机
    setx OLLAMA_HOST "127.0.0.1:11434"
    # 或者删掉
    reg delete "HKCU\\Environment" /v OLLAMA_HOST /f

（改完需要重开终端，并且重启 Ollama 服务才生效。）"""


def warn_on_env_misconfig():
    """如果环境变量 OLLAMA_HOST 和本项目配置的地址冲突，打印一条警告。

    只是警告，不影响运行——因为本项目已经显式指定了 base_url。
    """
    value = os.environ.get("OLLAMA_HOST")
    if not value:
        return None

    # 把 0.0.0.0:11434 / http://0.0.0.0:11434 之类的写法统一成 "host:port" 比较
    normalized = value.split("//")[-1]
    if normalized == f"{OLLAMA_HOST}:{OLLAMA_PORT}":
        return None

    message = _ENV_HOST_HINT.format(value=value)
    print("\n[警告] " + message.replace("\n", "\n        ") + "\n")
    return message


def ensure_ollama_ready(required_models):
    """自检。不通过就抛出 OllamaNotReady，消息里带处理步骤。

    required_models: 需要用到的模型名列表，例如 ["nomic-embed-text"]
    """
    if not _port_open():
        raise OllamaNotReady(
            _SERVICE_DOWN_HINT.format(host=OLLAMA_HOST, port=OLLAMA_PORT)
        )

    try:
        available = list_local_models()
    except Exception as exc:
        raise OllamaNotReady(
            f"Ollama 端口通了，但读取模型列表失败：{exc}\n"
            f"请检查 Ollama 是否正常运行。"
        ) from exc

    # Ollama 里的模型名带 tag（如 nomic-embed-text:latest），
    # 所以用"去掉 tag 后是否匹配"来判断
    available_base = {name.split(":")[0] for name in available}

    missing = [m for m in required_models if m.split(":")[0] not in available_base]
    if missing:
        raise OllamaNotReady(
            "Ollama 在运行，但缺少需要的模型：" + ", ".join(missing) + "\n\n"
            "已安装的模型：" + (", ".join(available) if available else "(空)") + "\n\n"
            "请执行：\n"
            + "\n".join(f"    ollama pull {m}" for m in missing)
        )


def ensure_for(need_llm=False):
    """按用途自检：只做检索就不需要拉生成模型。"""
    warn_on_env_misconfig()

    models = [config.EMBEDDING_MODEL]
    if need_llm:
        models.append(config.LLM_MODEL)
    ensure_ollama_ready(models)
