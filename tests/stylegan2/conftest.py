import random

import pytest
import torch


@pytest.fixture(autouse=True)
def seeds():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    random.seed(10)
