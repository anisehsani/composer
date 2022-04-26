# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from composer.core import data_spec
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.streaming import StreamingDatasetWriter
from composer.datasets.streaming.dataset import StreamingDataset


@pytest.fixture
def num_samples():
    return 4


@pytest.fixture
def image_size():
    return (16, 16)


@pytest.fixture
def pil_image_list(num_samples: int, image_size: Tuple[int, int]):
    return [Image.new(mode='RGB', size=image_size, color=(i, i, i)) for i in range(num_samples)]


@pytest.fixture
def pil_target_list(num_samples: int, image_size: Tuple[int, int]):
    return [Image.new(mode='L', size=image_size, color=i) for i in range(num_samples)]


@pytest.fixture
def correct_image_tensor(num_samples: int, image_size: Tuple[int, int]):
    return torch.arange(num_samples).expand(3, *image_size, -1).permute(3, 0, 1, 2)


@pytest.fixture
def scalar_target_list(num_samples: int):
    return np.arange(num_samples)


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
def test_mds_shuffling_pearson():
    n_shards = 8
    n_samples = 1000
    drop_last = True
    data_samples = np.arange(1000)
    shards = np.array_split(data_samples)

    fields = tuple('x')

    # have to figure out what this is in the Jenkins

    out_split_dir = "/?/train"

    with StreamingDatasetWriter(out_split_dir, fields) as out:
        for shard in shards:
            sample_generator = (i for i in range(shard))
            out.write_samples(sample_generator, use_tqdm=False, total=len(shard))

    remote = out_split_dir
    local = "/?/local"

    dataloader_hparams = DataLoaderHparams()  # Anis - need to init fields

    dataset = TestStreamingDataset(remote, local, True, transform=None, batch_size=10)
    return data_spec(dataloader=dataloader_hparams.initialize_object(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=True,
        collate_fn=None,
    ))


def test_mds_shuffling_pairs():
    pass
