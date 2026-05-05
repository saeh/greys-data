# data/dataset.py

import torch
from torch.utils.data import Dataset


class GreyhoundDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return {
            "idx": s["idx"],
            "datetime": s["datetime"],
            "track_id": s["track_id"],
            "distance": s["distance"],
            "dog_sequences": torch.tensor(s["dog_sequences"], dtype=torch.float32),
            "lengths": torch.tensor(s["lengths"], dtype=torch.long),
            "race_features": torch.tensor(s["race_features"], dtype=torch.float32),
            "dog_mask": torch.tensor(s["dog_mask"], dtype=torch.float32),
            "dog_ids": torch.tensor(s["dog_ids"], dtype=torch.long),
            "winner": torch.tensor(s["winner"], dtype=torch.long),
            "implied_probs": torch.tensor(s["implied_probs"], dtype=torch.float32),
        }
