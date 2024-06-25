
import pytest
import torch

from data.dataset import SeriesDataset, DynamicSeriesDataset, DynamicGraphSeriesDataset, DiscriminatorDataset


class TestSeriesDataset:
    @pytest.fixture
    def dataset(self):
        return SeriesDataset(
            sequences={
                0: [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]
                ],
                1: [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32]
                ],
                2: [
                    [33, 34, 35, 36],
                    [37, 38, 39, 40],
                    [41, 42, 43, 44],
                    [45, 46, 47, 48]
                ]
            },
            lookback=2
        )

    def test_len(self, dataset):
        assert len(dataset) == 6

    def test_getitem(self, dataset):
        x, y, i = dataset[0]
        assert x.shape == (2, 4)
        assert y.shape == (2,4)
        assert i == 0

    def test_getitem_last(self, dataset):
        x, y, i = dataset[-1]
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)
        assert i == 2



class TestDynamicSeriesDataset:
    @pytest.fixture
    def dataset(self):
        return DynamicSeriesDataset(
            sequences={
                0: [
                    torch.arange(4),
                    torch.arange(4,8),
                    torch.arange(8,12),
                    torch.arange(12,16)
                ],
                1: [
                    torch.arange(16,20),
                    torch.arange(20,24),
                    torch.arange(24,28),
                    torch.arange(28,32)
                ],
                2: [
                    torch.arange(32,36),
                    torch.arange(36,40),
                    torch.arange(40,44),
                    torch.arange(44,48)
                ]
            },
            lookback=2
        )

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem(self, dataset):
        x, v, y, i = dataset[0]
        assert x.shape == (2, 4)
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)
        assert i == 0

    def test_getitem_last(self, dataset):
        x, v, y, i = dataset[-1]
        assert x.shape == (2, 4)
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)
        assert i == 2



class TestDynamicGraphSeriesDataset:
    @pytest.fixture
    def dataset(self):
        return DynamicGraphSeriesDataset(
            sequences={
                0: [
                    torch.arange(4),
                    torch.arange(4,8),
                    torch.arange(8,12),
                    torch.arange(12,16)
                ],
                1: [
                    torch.arange(16,20),
                    torch.arange(20,24),
                    torch.arange(24,28),
                    torch.arange(28,32)
                ],
                2: [
                    torch.arange(32,36),
                    torch.arange(36,40),
                    torch.arange(40,44),
                    torch.arange(44,48)
                ]
            },
            adjacency_list={
                0: [
                    (0,[1,2],[]),
                    (1,[2],[1]),
                    (2,[1],[]),
                    (3,[1,2],[])
                ],
                1: [
                    (0,[0,2],[]),
                    (1,[2],[0]),
                    (2,[0],[]),
                    (3,[0,2],[])
                ],
                2: [
                    (0,[0,1],[]),
                    (1,[0,1],[]),
                    (2,[],[]),
                    (3,[0,1],[])
                ]
            },
            lookback=2
        )

    def test_len(self, dataset):
        assert len(dataset) == 1

    def test_getitem(self, dataset):
        x, v, a, y, i = dataset[0]
        assert x.shape == (2, 3, 4)
        assert x.shape == (2, 3, 4)
        assert a.shape == (2, 3, 3)
        assert y.shape == (2, 3, 4)
        assert i.shape == (2, 3)

    def test_getitem_last(self, dataset):
        x, v, a, y, i = dataset[-1]
        assert x.shape == (2, 3, 4)
        assert x.shape == (2, 3, 4)
        assert a.shape == (2, 3, 3)
        assert y.shape == (2, 3, 4)
        assert i.shape == (2, 3)



