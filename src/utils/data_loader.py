import torch
from torch.utils.data import DataLoader
from .dataset import TTS_Dataset  # 引入数据集类

def collate_fn(batch):
    """
    处理不同长度的文本数据
    """
    mel_spectrograms, texts, durations, speakers = zip(*batch)
    """
    # 打印形状，检查哪一维不匹配
    for i, mel in enumerate(mel_spectrograms):
        print(f"Sample {i}: Shape {mel.shape}")  
    mel_spectrograms = torch.nn.utils.rnn.pad_sequence(mel_spectrograms, batch_first=True, padding_value=0)
    durations = torch.tensor(durations, dtype=torch.float32)
    return mel_spectrograms, texts, durations, speakers
    """
    # 统一填充到最大时间步
    max_time_steps = max(mel.shape[-1] for mel in mel_spectrograms)  # 计算最长的时间步
    padded_mels = [torch.nn.functional.pad(mel, (0, max_time_steps - mel.shape[-1])) for mel in mel_spectrograms]  
    padded_mels = torch.stack(padded_mels)  # 转换为张量

    durations = torch.tensor(durations, dtype=torch.float32)

    return padded_mels, texts, durations, speakers

def get_dataloader(csv_path, mel_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = TTS_Dataset(csv_path, mel_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader
