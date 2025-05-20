import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from barhopping.config import GRANITE_MODEL

tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
model_em = AutoModel.from_pretrained(GRANITE_MODEL).eval()

# Adapter setup
ADAPTER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../adapter/adapter_model.pth")
)

class LinearAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Load adapter if available
adapter = None
with torch.no_grad():
    try:
        hidden_size = model_em.config.hidden_size
        adapter = LinearAdapter(hidden_size)

        if os.path.exists(ADAPTER_PATH):
            state_dict = torch.load(ADAPTER_PATH, map_location="cpu")
            adapter.load_state_dict(state_dict)
            adapter.eval()
        else:
            adapter = None
    except Exception as e:
        print(f"[Warning] Could not load adapter: {e}")
        adapter = None

def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        cls_embedding = model_em(**inputs)[0][:, 0]  # CLS token
        if adapter is not None:
            cls_embedding = adapter(cls_embedding)
        normalized = torch.nn.functional.normalize(cls_embedding, dim=1)
    return normalized
