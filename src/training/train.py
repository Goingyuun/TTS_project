# 文件路径: src/training/train.py

import os
from nemo.collections.tts.models import FastSpeech2Model
from nemo.core import Trainer

# 1. 加载 NVIDIA NeMo FastSpeech2 预训练模型
# 注意：预训练模型名称根据 NeMo 版本和模型发布可能有所不同，这里假设使用 "tts_en_fastspeech2"
model = FastSpeech2Model.from_pretrained("tts_en_fastspeech2")

# 2. 冻结所有参数，只微调适配层（例如 speaker_encoder）
for param in model.parameters():
    param.requires_grad = False
# 这里假设你只需要微调说话人嵌入层，实际项目中根据需求选择需要微调的部分
model.speaker_encoder.requires_grad = True

# 3. 准备数据加载器
# 假设你已经在 src/utils/data_loader.py 中实现了 get_dataloader 函数
from src.utils.data_loader import get_dataloader

# 指定数据索引文件和 Mel 特征目录路径
csv_path = "data/indices/index.csv"
mel_dir = "data/processed/mel"

# 创建训练和验证数据加载器
# 这里为了简单起见，使用相同的数据作为训练和验证数据，你可以根据需要分割数据
train_loader = get_dataloader(csv_path, mel_dir, batch_size=8, shuffle=True, num_workers=0)
val_loader = get_dataloader(csv_path, mel_dir, batch_size=8, shuffle=False, num_workers=0)

# 4. 使用 NeMo 的 Trainer 搭建训练框架
# 设置 devices=1（使用一块 GPU），并指定最大训练周期
trainer = Trainer(devices=1, accelerator="gpu", max_epochs=10)

# 开始训练
trainer.fit(model, train_loader, val_loader)

# 5. 训练结束后保存微调后的模型
os.makedirs("checkpoints", exist_ok=True)
model.save_to("checkpoints/finetuned_fastspeech2.nemo")
print("训练完成并保存模型到 checkpoints/finetuned_fastspeech2.nemo")

