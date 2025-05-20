import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .model import LinearAdapter
from .dataset import TripletDataset

def get_linear_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps-step) / float(max(1, total_steps-warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def train_linear_adapter(df, input_dim: int, batch_size: int = 32, epochs: int = 50, lr: float = 0.0001, warmup_steps: int = 100, margin: float = 0.5, device: str = 'cpu'):
    dataset = TripletDataset(df.anchor, df.positive, df.positive)
    val_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adapter = LinearAdapter(input_dim).to(device)
    adapter.train()

    optimizer = AdamW(adapter.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    hist_train_loss, hist_val_loss = [], []
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        for batch in train_loader:
            anchor, positive, negative = [x.to(device) for x in batch]
            
            out = adapter(anchor)
            loss = triplet_loss(out, positive, negative)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(adapter.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        hist_train_loss.append(train_loss/len(train_loader))

        adapter.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                anchor, positive, negative = [x.to(device) for x in batch]

                out = adapter(anchor)
                loss = triplet_loss(out, positive, negative)
                val_loss += loss.item()
        
        hist_val_loss.append(val_loss/len(val_loader))
        adapter.train()

    return adapter, hist_train_loss, hist_val_loss