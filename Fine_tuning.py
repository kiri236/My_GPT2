from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch

# 加载数据集
dataset = load_dataset("daily_dialog", name="dialog")

# 添加角色标记并拼接对话
def format_chat(examples):
    formatted = []
    for dialog in examples["dialog"]:
        # 每个对话样本是一个列表，包含多轮对话字符串
        chat = []
        for i, text in enumerate(dialog):
            role = "[用户]:" if i % 2 == 0 else "[助手]:"
            chat.append(f"{role} {text.strip()}")
        formatted.append(" ".join(chat))  # 拼接为单字符串
    return {"text": formatted}

# 应用格式转换
dataset = dataset.map(format_chat, batched=True, remove_columns=["dialog", "act"])

# 查看样例
print(dataset["train"][0]["text"])
# 输出示例: "[用户]: Hi ! [助手]: Hi . [用户]: How are you ? [助手]: Fine , thanks . ..."
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"additional_special_tokens": ["[用户]:", "[助手]:"]})  # 新增角色标记
tokenizer.pad_token = tokenizer.eos_token  # 设置填充符

def tokenize_function(examples):
    # 将文本转换为模型输入
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # 根据GPU显存调整
        padding="max_length"
    )
    # 创建标签（仅计算助手回复部分的损失）
    labels = tokenized["input_ids"].copy()
    # 将用户部分的标签设为 -100（损失计算时忽略）
    input_text = examples["text"][0]  # 取第一个样本的文本
    tokens = tokenizer.tokenize(input_text)
    user_turn = True
    for i, token in enumerate(tokens):
        if token == "[助手]:": user_turn = False
        if token == "[用户]:": user_turn = True
        if user_turn:
            labels[i] = -100  # 掩码用户部分的损失
    tokenized["labels"] = labels
    return tokenized

# 应用分词
tokenized_dataset = dataset.map(tokenize_function, batched=False, batch_size=1)  # batch_size=1避免跨样本处理错误

# 转换为PyTorch格式
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from transformers import GPT2LMHeadModel

# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 调整模型嵌入层（适配新增的特殊标记）
model.resize_token_embeddings(len(tokenizer))
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./gpt2-chatbot",
    per_device_train_batch_size=4,  # 根据GPU调整（如RTX 3090可设为8）
    num_train_epochs=3,
    learning_rate=3e-5,            # 比预训练更小的学习率
    logging_steps=50,
    save_steps=500,
    fp16=False,                     # 开启混合精度训练（NVIDIA GPU需支持）
    gradient_accumulation_steps=2, # 显存不足时使用梯度累积
    evaluation_strategy="no",      # daily_dialog无验证集
)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        return outputs.loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)
trainer.train()

# 保存最终模型
model.save_pretrained("./gpt2-chatbot-final")
tokenizer.save_pretrained("./gpt2-chatbot-final")