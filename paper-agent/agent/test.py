def _clean_json(raw: str) -> str:
    """把大模型返回的"脏"文本，洗成一段能交给 json.loads 的文本。

    要对付两种脏法：
      1. 被 ```json ... ``` 包起来的
      2. 前后带废话的，比如：好的，我的判断是：{"enough": true, ...}
    """
    # TODO 1：先删掉代码块围栏 —— 把 "```json" 和 "```" 这两个子串去掉。
    #   提示：字符串的 .replace(旧, 新) 方法，新 传空串 "" 就是删除。
    #   注意大小写和顺序：先删 "```json"（更长更具体），再删剩下的 "```"。
    #   结果存回 raw 里。
    #   大模型爱用 Markdown 代码块来包 JSON。Markdown 里，代码块的写法就是三个反引号开头、三个反引号结尾：
    #   ```json
    #   {"enough": true, "next_query": null}
    #   ```
    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")

    # TODO 2：再截出最外层的花括号 —— 只要【第一个 {】到【最后一个 }】之间。
    #   提示：
    #     raw.find("{")    → 找第一个 {，返回下标数字
    #     raw.rfind("}")   → 从右边开始找，返回最后一个 } 的下标
    #     raw[左:右+1]     → 切片。为什么要 +1？切片"包头不包尾"。
    #   把切出来的结果 return 回去。
    index1 = raw.find("{")
    index2 = raw.rfind("}")
    # 左闭右开
    raw = raw[index1 : index2 + 1]
    return raw


string = "```json{123}```"
print(_clean_json(string))
