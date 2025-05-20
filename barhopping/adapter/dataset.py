import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple

class TripletDataset(Dataset):
    def __init__(self, anchor_df: pd.Series, positive_df: pd.Series, negative_df: pd.Series):
        """
        Dataset for triplet inputs: anchor, positive, negative samples.
        
        Args:
            anchor_df (pd.Series): Series of JSON strings for anchor vectors.
            positive_df (pd.Series): Series of JSON strings for positive vectors.
            negative_df (pd.Series): Series of JSON strings for negative vectors.
        """
        self.anchor = anchor_df
        self.positive = positive_df
        self.negative = negative_df

    def __len__(self) -> int:
        return len(self.anchor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return a triplet (anchor, positive, negative) as tensors.

        Negative sample is randomly chosen.

        Args:
            idx (int): Index for anchor and positive samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: anchor, positive, negative tensors.
        """
        a_vec = torch.tensor(json.loads(self.anchor.iloc[idx]), dtype=torch.float32)
        p_vec = torch.tensor(json.loads(self.positive.iloc[idx]), dtype=torch.float32)
        n_vec = torch.tensor(json.loads(self.negative.sample(1).iloc[0]), dtype=torch.float32)
        return a_vec, p_vec, n_vec