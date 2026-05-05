import polars as pl
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm

MAX_DOGS = 10


def load_parquet_data(path):
    df = pl.read_parquet(path)

    df = df.sort(["datetime_utc", "idx", "racebox"])

    df = df.with_columns(
        pl.col("win_bsp").fill_null(1.0).alias("bsp_raw"),
        np.log(pl.col("win_bsp").fill_null(1.0)).alias("log_bsp"),
    )

    df = df.with_columns((pl.col("placenum") == 1).cast(pl.Int32).alias("is_winner"))

    speed_mean = df["speedc"].mean()
    speed_std = df["speedc"].std() + 1e-8
    df = df.with_columns(((pl.col("speedc") - speed_mean) / speed_std).alias("speedc"))

    log_bsp_mean = df["log_bsp"].mean()
    log_bsp_std = df["log_bsp"].std() + 1e-8
    df = df.with_columns(
        ((pl.col("log_bsp") - log_bsp_mean) / log_bsp_std).alias("log_bsp")
    )

    splitmargin_mean = df["splitmargin"].mean()
    splitmargin_std = df["splitmargin"].std() + 1e-8
    df = df.with_columns(
        ((pl.col("splitmargin") - splitmargin_mean) / splitmargin_std).alias(
            "splitmargin"
        )
    )

    for col in ["pir1", "pir2", "pir3"]:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df = df.with_columns(((pl.col(col) - mean) / std).alias(col))

    for col in ["margin1", "margin2"]:
        df = df.with_columns(np.log(pl.col(col) + 1.0).alias(f"log_{col}"))

    speed_mean = df["speedc"].mean()
    speed_std = df["speedc"].std() + 1e-8
    df = df.with_columns(((pl.col("speedc") - speed_mean) / speed_std).alias("speedc"))

    weight_mean = df["weight"].mean()
    weight_std = df["weight"].std() + 1e-8
    df = df.with_columns(
        ((pl.col("weight") - weight_mean) / weight_std).alias("weight")
    )

    df = df.with_columns((pl.col("distance") / 1000.0).alias("distance"))

    for i in range(1, 4):
        df = df.with_columns(
            np.log(pl.col(f"prizemoney{i}") + 1.0)
            .fill_null(0.0)
            .alias(f"log_prizemoney{i}")
        )

    df = df.with_columns(
        pl.col("runs_tot").cast(pl.Float32).alias("runs_tot_f"),
        pl.col("wins_tot").cast(pl.Float32).alias("wins_tot_f"),
        pl.col("secs_tot").cast(pl.Float32).alias("secs_tot_f"),
        pl.col("thir_tot").cast(pl.Float32).alias("thir_tot_f"),
        pl.col("runs_ttd").cast(pl.Float32).alias("runs_ttd_f"),
        pl.col("wins_ttd").cast(pl.Float32).alias("wins_ttd_f"),
        pl.col("secs_ttd").cast(pl.Float32).alias("secs_ttd_f"),
        pl.col("thir_ttd").cast(pl.Float32).alias("thir_ttd_f"),
        pl.col("BT_NUM").cast(pl.Float32).alias("best_time_f"),
        pl.col("speedc").cast(pl.Float32).alias("speedc_f"),
        pl.col("weight").cast(pl.Float32).alias("weight_f"),
        pl.col("dogprize").cast(pl.Float32).alias("dogprize_f"),
    )

    df = df.with_columns(
        pl.when(pl.col("sex") == "D").then(1.0).otherwise(0.0).alias("sex_d"),
        pl.when(pl.col("sex") == "M").then(1.0).otherwise(0.0).alias("sex_m"),
    )

    return df


def build_dog_histories(df, max_len=20):
    """Assumes the DF is sorted by date"""
    dog_histories = defaultdict(lambda: deque(maxlen=max_len))
    history_per_row = []

    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="  Building histories"):
        dog = row["dogname"]

        history_per_row.append(list(dog_histories[dog]))

        if len(dog_histories[dog]) > 0:
            last_date = dog_histories[dog][-1]["date"]
            delta_days = (row["datetime_utc"] - last_date).total_seconds() / (3600 * 24)
        else:
            delta_days = 0.0

        history_entry = {
            "date": row["datetime_utc"],
            "racebox": row["racebox"],
            "box": row["box"],
            "distance": row["distance"],
            "time_since_last": delta_days,
            "weight": row["weight_f"],
            "dogprize": row["dogprize_f"],
            "log_prizemoney1": row["log_prizemoney1"],
            "log_prizemoney2": row["log_prizemoney2"],
            "log_prizemoney3": row["log_prizemoney3"],
            "speed": row["speedc_f"],
            "log_bsp": row["log_bsp"],
            "placenum": min(row["placenum"], 8),
            "log_margin1": row["log_margin1"],
            "log_margin2": row["log_margin2"],
            "splitmargin": row["splitmargin"],
            "pir1": row["pir1"],
            "pir2": row["pir2"],
            "pir3": row["pir3"],
        }

        dog_histories[dog].append(history_entry)

    df = df.with_columns(pl.Series("history", history_per_row))
    return df


def encode_history(history, feature_keys):
    if len(history) == 0:
        return np.zeros((1, len(feature_keys)), dtype=np.float32), 1

    seq = [[h[k] for k in feature_keys] for h in history]
    return np.array(seq, dtype=np.float32), len(seq)


def build_samples(
    df, max_seq_len=20, dog_name_to_id=None, track_name_to_id=None
):
    samples = []

    feature_keys = [
        "speed",
        "box",
        "distance",
        "time_since_last",
        "weight",
        "dogprize",
        "log_prizemoney1",
        "log_prizemoney2",
        "log_prizemoney3",
        "placenum",
        "log_margin1",
        "log_margin2",
        "splitmargin",
        "pir1",
        "pir2",
        "pir3",
    ]

    feat_dim = len(feature_keys)

    grouped = df.group_by("idx", maintain_order=True)

    for race_id, group in grouped:
        raceboxes = group["racebox"].to_numpy()
        bsp_raw = group["bsp_raw"].to_numpy()
        distances = group["distance"].to_numpy()
        histories = group["history"].to_list()
        dognames = group["dogname"].to_list()
        prizemoney1 = group["log_prizemoney1"].to_numpy()
        prizemoney2 = group["log_prizemoney2"].to_numpy()
        prizemoney3 = group["log_prizemoney3"].to_numpy()
        placings = group["placenum"].to_numpy()

        # NOTE: Sorting by box ensures consistent ordering for training.
        # For inference data this is not known ahead of time so we will
        # sort by box later.
        sorted_idx = np.argsort(raceboxes)
        num_dogs = len(sorted_idx)

        race_datetime = group["datetime_utc"][0]
        track_name = group["track"][0]
        track_id = track_name_to_id.get(f"__track__{track_name}", 0)

        dog_sequences = []
        lengths = []
        race_features = []
        dog_ids = []

        for i in sorted_idx:
            history = histories[i][-max_seq_len:]
            if len(history) == 0:
                seq = np.zeros((1, feat_dim), dtype=np.float32)
                length = 1
            else:
                seq = np.array(
                    [[h[k] for k in feature_keys] for h in history], dtype=np.float32
                )
                length = len(seq)
            dog_sequences.append(seq)
            lengths.append(length)
            race_features.append(
                [
                    raceboxes[i],
                    distances[i],
                    length,
                    prizemoney1[i],
                    prizemoney2[i],
                    prizemoney3[i],
                ]
            )
            # For dogs not seen in training, we assign an ID of 0 which the model can learn to handle as "unknown dog".
            dog_ids.append(dog_name_to_id[dognames[i]] if dog_name_to_id and dognames[i] in dog_name_to_id else 0)

        max_len = max(lengths)

        padded = []
        for seq in dog_sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                seq = np.vstack([np.zeros((pad_len, feat_dim), dtype=np.float32), seq])
            padded.append(seq)

        while len(padded) < MAX_DOGS:
            padded.append(np.zeros((max_len, feat_dim), dtype=np.float32))
            lengths.append(1)
            race_features.append([0, 0, 0, 0, 0, 0])
            dog_ids.append(0)

        # Shape: (num_dogs=8[padded], seq_len=max_len, feat_dim=16)
        dog_sequences = np.stack(padded).astype(np.float32)
        # Shape: (num_dogs=8[padded], feat_dim=6)
        race_features = np.array(race_features, dtype=np.float32)
        lengths = np.array(lengths, dtype=np.int64)
        dog_ids = np.array(dog_ids, dtype=np.int64)

        dog_mask = np.zeros(MAX_DOGS, dtype=np.float32)
        dog_mask[:num_dogs] = 1.0

        sample = {
            "idx": race_id,
            "datetime": race_datetime,
            "track_id": track_id,
            "distance": distances[0],
            "dog_sequences": dog_sequences,
            "lengths": lengths,
            "race_features": race_features,
            "dog_mask": dog_mask,
            "dog_ids": dog_ids,
        }

        implied = 1.0 / bsp_raw[sorted_idx]
        implied = implied / implied.sum()
        padded_implied = np.zeros(MAX_DOGS, dtype=np.float32)
        padded_implied[:num_dogs] = implied

        try:
            winner_idx = np.where(placings == 1)[0][0]
            sorted_winner_idx = np.where(sorted_idx == winner_idx)[0][0]
            sample["winner"] = int(sorted_winner_idx)

        except IndexError:
            # Handle case where no winner is found
            sorted_winner_idx = -1
            sample["winner"] = sorted_winner_idx

        sample["implied_probs"] = padded_implied
        samples.append(sample)

    return samples


def build_dog_name_mapping(df: pl.DataFrame) -> dict[str, int]:
    all_dogs: pl.Series = df["dogname"].unique().cast(dtype=str)
    dog_name_to_id: dict[str, int] = {
        name: idx + 1 for idx, name in enumerate(sorted(all_dogs))
    }
    return dog_name_to_id


def build_track_name_mapping(df):
    all_tracks = df["track"].unique()
    track_name_to_id = {
        f"__track__{name}": idx + 1 for idx, name in enumerate(sorted(all_tracks))
    }
    return track_name_to_id
