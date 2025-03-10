# 文件路径: src/training/train.py

import os
import torch
import soundfile as sf
from nemo.collections.tts.models import FastSpeech2HifiGanE2EModel
from nemo.core import Trainer

# 1. 加载 NVIDIA NeMo FastSpeech2HifiGanE2E 预训练模型
model = FastSpeech2HifiGanE2EModel.from_pretrained("tts_en_e2e_fastspeech2hifigan")

# 2. 冻结除变长自适应器（variance adaptor）之外的参数
for param in model.parameters():
    param.requires_grad = False
for param in model.variance_adaptor.parameters():
    param.requires_grad = True  # 仅微调 variance adaptor，以适应个性化语音特征

# 3. 准备数据加载器
# 由于 FastSpeech2HifiGanE2E 不使用 Mel 频谱，而是直接生成波形，我们需要提供文本数据
from src.utils.data_loader import get_text_dataloader

csv_path = "data/indices/index.csv"  # 你的文本数据索引文件
train_loader = get_text_dataloader(csv_path, batch_size=8, shuffle=True)
val_loader = get_text_dataloader(csv_path, batch_size=8, shuffle=False)

# 4. 使用 NeMo Trainer 进行训练
trainer = Trainer(devices=1, accelerator="gpu", max_epochs=10)

# 开始训练
trainer.fit(model, train_loader, val_loader)

# 5. 训练结束后保存微调后的模型
os.makedirs("checkpoints", exist_ok=True)
model.save_to("checkpoints/finetuned_fastspeech2hifigan.nemo")
print("训练完成，模型已保存至 checkpoints/finetuned_fastspeech2hifigan.nemo")

# 6. 进行推理测试
test_text = "Hello, this is a personalized voice synthesis test."
tokens = model.parse(test_text)  # 解析文本为 token
audio = model.convert_text_to_waveform(tokens=tokens)  # 直接生成波形

# 保存测试音频
sf.write("test_speech.wav", audio.to('cpu').numpy(), 22050)
print("测试音频已保存为 test_speech.wav")
mo")

