import torch
from torch.utils.data import DataLoader
from .dataset import TTS_Dataset  # 引入数据集类

def collate_fn(batch):
    """
    处理不同长度的文本数据
    """
    texts, durations, speakers = zip(*batch)
    
    # 对文本进行 padding (如果需要的话，通常是处理词语长度不一的情况)
    max_text_len = max(len(text) for text in texts)
    padded_texts = [text + ['<pad>'] * (max_text_len - len(text)) for text in texts]  # 填充文本到最大长度
    padded_texts = torch.tensor(padded_texts, dtype=torch.long)

    durations = torch.tensor(durations, dtype=torch.float32)
    
    return padded_texts, durations, speakers

def get_dataloader(csv_path, mel_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = TTS_Dataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader
