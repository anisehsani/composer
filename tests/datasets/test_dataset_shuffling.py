# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from composer.core import data_spec
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.streaming import StreamingDatasetWriter
from composer.datasets.streaming.dataset import StreamingDataset


class TestStreamingDataset(StreamingDataset):

    def decode_int(self, data: bytes) -> Any:
        return np.frombuffer(data, np.int64)[0]

    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 trainsform: Optional[Callable] = None,
                 device_batch_size: int = 0) -> None:
        decoders = {'x': self.decode_int}
        super().__init__(remote, local, shuffle, decoders, device_batch_size)
        # transform?

    def __getitem__(self, idx: int) -> Tuple[Any]:
        obj = super().__getitem__(idx)
        x = obj['x']
        return x

        # Anis - Maybe I need a default transform field here?


# mark parameterize by n_shards/n_samples/drop_last
def test_mds_shuffling_pearson(tmpdir: pathlib.Path):
    n_shards = 8
    n_samples = 1000
    drop_last = True
    data_samples = np.arange(1, n_samples + 1)
    shards = np.array_split(data_samples, n_shards)

    batch_size = 1

    fields = tuple('x')

    # have to figure out what this is in the Jenkins

    out_split_dir = os.path.join(tmpdir, "remote", "train")
    local_split_dir = os.path.join(tmpdir, "local", "train")
    os.makedirs(out_split_dir)
    os.makedirs(local_split_dir)

    with StreamingDatasetWriter(out_split_dir, fields) as out:
        for shard in shards:
            sample_generator = (i for i in shard)
            out.write_samples(sample_generator, use_tqdm=False, total=len(shard))


# Anis - need to init fields

# add num_workers later
    dataset = TestStreamingDataset(out_split_dir, local_split_dir, True, transform=None, batch_size=batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=drop_last)

    returned_data = [int(x) for x in dataloader]

    rank_correlation = np.cov(np.array(returned_data), data_samples)
    assert abs(rank_correlation) < 0.5  # change threshold later


def test_mds_shuffling_pairs():
    pass
