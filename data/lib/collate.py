# collate.py
import torch
import numpy as np


def collate_fn(batch):
    # Find max seq_len in batch
    max_seq_len = max(sample["dog_sequences"].shape[1] for sample in batch)

    num_dogs = 8
    feat_dim = batch[0]["dog_sequences"].shape[2]

    idxs = []
    datetimes = []
    track_ids = []
    distances = []
    dog_sequences = []
    lengths = []
    race_features = []
    dog_masks = []
    dog_ids = []
    winners = []
    implied_probs = []

    for sample in batch:
        seq = sample["dog_sequences"]  # [8, seq_len, feat_dim]
        seq_len = seq.shape[1]

        # Pad time dimension (left-pad)
        if seq_len < max_seq_len:
            pad = np.zeros(
                (num_dogs, max_seq_len - seq_len, feat_dim), dtype=np.float32
            )
            seq = np.concatenate([pad, seq], axis=1)

        idxs.append(sample["idx"])
        datetimes.append(sample["datetime"])
        track_ids.append(sample["track_id"])
        distances.append(sample["distance"])
        dog_sequences.append(seq)
        lengths.append(sample["lengths"])
        race_features.append(sample["race_features"])
        dog_masks.append(sample["dog_mask"])
        dog_ids.append(sample["dog_ids"])
        winners.append(sample["winner"])
        implied_probs.append(sample["implied_probs"])

    return {
        "idxs": idxs,
        "datetimes": datetimes,
        "track_ids": track_ids,
        "distances": distances,
        "dog_sequences": torch.tensor(np.stack(dog_sequences), dtype=torch.float32),
        "lengths": torch.tensor(np.stack(lengths), dtype=torch.long),
        "race_features": torch.tensor(np.stack(race_features), dtype=torch.float32),
        "dog_mask": torch.tensor(np.stack(dog_masks), dtype=torch.float32),
        "dog_ids": torch.tensor(np.stack(dog_ids), dtype=torch.long),
        "winner": torch.tensor(winners, dtype=torch.long),
        "implied_probs": torch.tensor(np.stack(implied_probs), dtype=torch.float32),
    }
