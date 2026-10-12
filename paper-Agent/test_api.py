import api
from fastapi.testclient import TestClient


def boom(*a, **kw):
    raise RuntimeError("模拟：DeepSeek 挂了")


api.graph.invoke = boom  # ← 只在这个测试进程里替换

if __name__ == "__main__":
    # with 进去 → 会真的跑一遍 lifespan（加载引擎）
    with TestClient(api.app, raise_server_exceptions=False) as c:
        r = c.post("/ask", json={"question": "随便问"})

    print()
    print("=" * 56)
    print("客户端收到：")
    print("  HTTP 状态码 :", r.status_code)
    print("  响应体      :", r.json())
    print("=" * 56)
