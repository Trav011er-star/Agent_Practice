import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Hugging Face Transformers 是一个强大的开源库，
# 它提供了标准化的接口来加载和使用数以万计的预训练模型。

# AutoTokenizer 分词器
# AutoModelForCausalLM 加载因果语言模型（用上一个词预测下一个词）

# 加载预训练模型和分词器
model_name = "Qwen/Qwen1.5-0.5B-chat"

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print("模型和分词器加载完成。")

# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请给我讲解一下Transformer模型。"},
]
# System：（系统指令）
# 你是一个有帮助的助手。

# User：（用户消息）
# 你好，请介绍你自己。

# Assistant：（助手回应）
# 模型接下来要做的，就是预测 Assistant 后面的内容。

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    # 这一步先只生成格式化后的字符串，不马上变成 Token ID。
    tokenize=False,
    add_generation_prompt=True,
)

print("\n格式化后的输入文本:")
print(text)

# 编码输入文本
# 文本
# ↓
# 切成 Token
# ↓
# 查词表
# ↓
# 转换成 Token ID
# ↓
# PyTorch Tensor
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# model_inputs = {
#     "input_ids": tensor([[151644, 8948, 198, ...]]),
#     "attention_mask": tensor([[1, 1, 1, ...]])
# }
print("编码后的输入文本:")
print(model_inputs)

# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

print("\n生成的Token ID:")
print(generated_ids)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

print("\n生成的Token ID (去除输入部分):")
print(generated_ids)

# Token ID → Token -> 组合成文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(response)
