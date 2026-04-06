"""Penn Treebank word-level data loader with auto-download."""

import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader


PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
PTB_FILES = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]


def _download_ptb(data_dir):
    """Download PTB files if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    for fname in PTB_FILES:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            url = PTB_URL + fname
            print(f"[ptb] Downloading {url}")
            urllib.request.urlretrieve(url, fpath)


class Vocabulary:
    """Simple word-level vocabulary built from a text file."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_file(self, path):
        with open(path, "r") as f:
            for line in f:
                for word in line.strip().split() + ["<eos>"]:
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.idx2word)
                        self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, path):
        ids = []
        with open(path, "r") as f:
            for line in f:
                for word in line.strip().split() + ["<eos>"]:
                    ids.append(self.word2idx[word])
        return torch.tensor(ids, dtype=torch.long)


class PTBDataset(Dataset):
    """Chops a flat token tensor into fixed-length sequences."""

    def __init__(self, data, block_size=128):
        self.block_size = block_size
        # Trim to a multiple of block_size
        n_sequences = len(data) // block_size
        self.data = data[: n_sequences * block_size].view(n_sequences, block_size)

    def __len__(self):
        return len(self.data) - 1  # need next token for targets

    def __getitem__(self, idx):
        # Input: sequence at idx, Target: sequence at idx+1
        # This gives overlapping sequences shifted by one block
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y


def get_ptb_loaders(batch_size=128, block_size=128, num_workers=2, data_dir="./data_cache/ptb"):
    """Return (train_loader, val_loader, test_loader, vocab_size) for PTB.

    Args:
        batch_size: Batch size for all loaders.
        block_size: Context length (sequence length).
        num_workers: DataLoader workers. Set to 0 if you hit macOS multiprocessing issues.
        data_dir: Directory to download/cache PTB data.
    """
    _download_ptb(data_dir)

    vocab = Vocabulary()
    # Build vocab from train set only
    vocab.add_file(os.path.join(data_dir, "ptb.train.txt"))
    # But also add any words from val/test to avoid OOV
    vocab.add_file(os.path.join(data_dir, "ptb.valid.txt"))
    vocab.add_file(os.path.join(data_dir, "ptb.test.txt"))

    train_data = vocab.encode(os.path.join(data_dir, "ptb.train.txt"))
    val_data = vocab.encode(os.path.join(data_dir, "ptb.valid.txt"))
    test_data = vocab.encode(os.path.join(data_dir, "ptb.test.txt"))

    train_set = PTBDataset(train_data, block_size)
    val_set = PTBDataset(val_data, block_size)
    test_set = PTBDataset(test_data, block_size)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, len(vocab)
