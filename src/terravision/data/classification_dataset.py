"""PyTorch Dataset for EuroSAT, driven by the split CSV from eurosat.prepare_eurosat."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class EuroSATDataset(Dataset):
    """Returns ``(image, label)`` for one split.

    split_csv:  CSV with columns path, class_name, label, split
    class_root: folder the ``path`` column is relative to
    split:      "train", "val" or "test"
    """

    def __init__(self, split_csv: str | Path, class_root: str | Path, split: str,
                 transform: Callable | None = None):
        df = pd.read_csv(split_csv)
        if split not in set(df["split"]):
            raise ValueError(f"Split '{split}' not found in {split_csv}")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.class_root = Path(class_root)
        self.transform = transform
        # label -> name, taken from the CSV (the scan defined the label order).
        pairs = df[["label", "class_name"]].drop_duplicates().sort_values("label")
        self.classes: list[str] = pairs["class_name"].tolist()
        if pairs["label"].tolist() != list(range(len(self.classes))):
            raise ValueError("Labels in the split CSV are not 0..N-1")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(self.class_root / row["path"]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
