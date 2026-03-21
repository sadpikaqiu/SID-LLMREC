import numpy as np

from gnprsid.retrieval.similarity import build_candidate_mask


def test_build_candidate_mask_filters_future_and_self():
    train_ids = np.array(["train-1", "train-2", "train-3"])
    train_times = np.array([1.0, 2.0, 3.0])
    mask = build_candidate_mask(train_ids, train_times, "train-2", 2.5, "train")
    assert mask.tolist() == [True, False, False]
