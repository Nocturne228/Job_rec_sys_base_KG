import torch
from scipy import sparse

from src.data import DataLoader, generate_mock_data
from src.ranking.skill_coverage import SkillCoverageCalculator
from src.recall import LightGCN
from src.recall.lightgcn import prepare_adj_matrix
from src.utils.training import evaluate_model, train_full_pipeline


def test_mock_generation_and_per_user_holdout_are_reproducible():
    first = generate_mock_data(8, 12, seed=19)
    second = generate_mock_data(8, 12, seed=19)
    assert first.model_dump() == second.model_dump()

    loader = DataLoader(first, test_ratio=0.2, random_seed=19)
    assert loader.train_R.nnz + loader.test_R.nnz == loader.R.nnz
    assert all(loader.train_R[user_idx].nnz > 0 for user_idx in loader.test_users)


def test_gat_coverage_respects_proficiency_level():
    class FixedWeighter:
        def get_skill_weight(self, skill_id):
            return 1.0

    calculator = SkillCoverageCalculator(gat_weighter=FixedWeighter())
    result = calculator.calculate_coverage(
        {"python": "beginner"}, {"python": "advanced"}
    )
    assert result["coverage_score"] == 0.0
    assert result["gat_coverage_score"] == 0.0


def test_heldout_evaluation_masks_training_interactions():
    model = LightGCN(n_users=1, n_items=3, embedding_dim=2, n_layers=0)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.item_embedding.weight.copy_(
            torch.tensor([[10.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )

    adjacency = prepare_adj_matrix(sparse.eye(4, dtype="float32"))
    train_r = torch.tensor([[1.0, 0.0, 0.0]])
    test_r = torch.tensor([[0.0, 1.0, 0.0]])

    unmasked = evaluate_model(model, test_r, adjacency, k_values=[1])
    masked = evaluate_model(model, test_r, adjacency, k_values=[1], train_R=train_r)
    assert unmasked["recall@1"] == 0.0
    assert masked["recall@1"] == 1.0


def test_offline_model_initialization_and_training_are_reproducible():
    data = generate_mock_data(6, 10, seed=42)
    config = {
        "lightgcn_embedding_dim": 8,
        "lightgcn_n_layers": 1,
        "lightgcn_dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "n_epochs": 1,
        "device": "cpu",
    }
    first = train_full_pipeline(DataLoader(data, random_seed=42), config=config)[
        "model"
    ]
    second = train_full_pipeline(DataLoader(data, random_seed=42), config=config)[
        "model"
    ]
    for first_value, second_value in zip(
        first.state_dict().values(), second.state_dict().values()
    ):
        assert torch.equal(first_value, second_value)
