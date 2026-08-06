import torch

from src.data import DataLoader, generate_mock_data
from src.utils.training import evaluate_embeddings, sample_unobserved_negatives


def test_per_user_split_is_reproducible_and_training_only():
    data = generate_mock_data(8, 12, seed=19)
    first = DataLoader(data, random_seed=19)
    second = DataLoader(data, random_seed=19)
    assert (first.train_R != second.train_R).nnz == 0
    assert (first.test_R != second.test_R).nnz == 0
    assert first.train_R.multiply(first.test_R).nnz == 0
    assert first.train_R.nnz + first.test_R.nnz == first.R.nnz
    assert all(first.train_R[user].nnz > 0 for user in first.test_users)
    timestamps = {(row.user_id, row.job_id): row.timestamp for row in data.interactions}
    for user_idx in first.test_users:
        user_id = first.idx_to_user_id[user_idx]
        train_times = [
            timestamps[(user_id, first.idx_to_job_id[item])]
            for item in first.train_R[user_idx].indices
        ]
        test_times = [
            timestamps[(user_id, first.idx_to_job_id[item])]
            for item in first.test_R[user_idx].indices
        ]
        assert max(train_times) <= min(test_times)


def test_negative_sampling_never_returns_training_seen_item():
    train = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    negatives, positions = sample_unobserved_negatives(train, torch.tensor([0, 1]))
    assert positions.tolist() == [0]
    assert negatives.tolist() == [1]
    assert train[0, negatives[0]] == 0


def test_evaluation_masks_training_seen_items():
    users = torch.tensor([[1.0, 0.0]])
    items = torch.tensor([[10.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    train = torch.tensor([[1.0, 0.0, 0.0]])
    test = torch.tensor([[0.0, 1.0, 0.0]])
    metrics = evaluate_embeddings(users, items, test, train, k=1)
    assert metrics["recall@10"] == 1.0
