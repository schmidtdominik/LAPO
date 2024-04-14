import random
from collections.abc import Generator
from pathlib import Path

import doy
import numpy as np
import paths
import torch
from config import ADD_TIME_HORIZON, DEVICE
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader

TRAIN_CHUNK_LEN = 32_768
TEST_CHUNK_LEN = 4096


def _create_tensordict(length: int, obs_depth) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.zeros(length, obs_depth, 64, 64, dtype=torch.uint8),
            "ta": torch.zeros(length, dtype=torch.long),
            "done": torch.zeros(length, dtype=torch.bool),
            "rewards": torch.zeros(length),
            "ep_returns": torch.zeros(length),
            "values": torch.zeros(length),
        },
        batch_size=length,
        device="cpu",
    )


def _unfold_td(td: TensorDictBase, seq_len: int, unfold_step: int = 1):
    """
    Unfolds the given TensorDict along the time dimension.
    The unfolded TensorDict shares its underlying storage with the original TensorDict.
    """
    res_batch_size = (td.batch_size[0] - seq_len + 1,)
    td = td.apply(
        lambda x: x.unfold(0, seq_len, unfold_step).movedim(-1, 1),
        batch_size=res_batch_size,
    )
    return td


def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    assert not torch.is_floating_point(obs)
    return obs.float() / 255 - 0.5


class DataStager:
    def __init__(
        self,
        files: list[Path],
        chunk_len: int,
        obs_depth: int = 3,
        seq_len: int = 2,
    ) -> None:

        self.seq_len = seq_len
        self.td: TensorDict = None  # type: ignore
        self.obs_depth = obs_depth
        self.files = files
        self.chunk_len = chunk_len
        random.shuffle(self.files)

        self.td = _create_tensordict(self.chunk_len * len(self.files), self.obs_depth)
        self.td_unfolded = _unfold_td(self.td, self.seq_len, 1)
        self._load()

    def _load(self):
        for i, path in enumerate(self.files):
            self._load_chunk(path, i)

    def _load_chunk(self, path: Path, i: int):
        data = np.load(path)
        for k in self.td.keys():
            v = torch.from_numpy(data[k])
            if k == "obs":
                v = v.permute(0, 3, 1, 2)
            assert len(v) == self.chunk_len, v.shape
            self.td[k][i * self.chunk_len : (i + 1) * self.chunk_len] = v

    def get_iter(
        self,
        batch_size: int,
        device=DEVICE,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        dataloader = DataLoader(
            self.td_unfolded,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: x,
        )

        while True:
            for batch in dataloader:
                batch = batch.to(device)
                batch["obs"] = normalize_obs(batch["obs"])
                yield batch


def _load(env_name: str, test: bool):
    chunk_len = TEST_CHUNK_LEN if test else TRAIN_CHUNK_LEN
    return DataStager(
        files=paths.get_expert_data(env_name, test),
        chunk_len=chunk_len,
        seq_len=2 + ADD_TIME_HORIZON,
    )


def load(env_name: str) -> tuple[DataStager, DataStager]:
    with doy.status(f"Loading expert data for env: {env_name}"):
        return _load(env_name, test=False), _load(env_name, test=True)
