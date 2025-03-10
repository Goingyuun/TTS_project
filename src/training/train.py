# 文件路径: src/training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.utils.data_loader import get_dataloader  


# 定义一个简化版的 Dummy 模型（模拟 FastSpeech2 的数据流）
class DummyFastSpeech2(nn.Module):
    def __init__(self, n_mels=80):
        super(DummyFastSpeech2, self).__init__()
        # 这里我们用一个简单的线性层来模拟处理
        self.linear = nn.Linear(n_mels, n_mels)

    def forward(self, x):
        # 输入 x: (batch, time_steps, n_mels)
        # 直接对最后一个维度进行线性变换
        out = self.linear(x)
        return out


def train_model():
    # 超参数设置
    num_epochs = 5
    batch_size = 16
    learning_rate = 0.001
    n_mels = 80

    # 数据路径
    csv_path = "data/indices/index.csv"
    mel_dir = "data/processed/mel"

    # 获取数据加载器
    dataloader = get_dataloader(
        csv_path, mel_dir, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # 模型、损失函数和优化器初始化
    model = DummyFastSpeech2(n_mels=n_mels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []

    # 基础训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            # 解包数据：mel_batch 的形状为 (batch, time_steps, n_mels)
            mel_batch, text_batch, duration_batch, speaker_batch = batch
            mel_batch = mel_batch.to(device)

            optimizer.zero_grad()
            outputs = model(mel_batch)
            loss = criterion(outputs, mel_batch)  # 以输入作为目标，验证数据流和损失计算
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 绘制训练损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("training_loss_curve.png")
    plt.show()

    # 保存模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/dummy_fastspeech2.pth")
    print("Training completed and model saved.")


if __name__ == "__main__":
    train_model()
