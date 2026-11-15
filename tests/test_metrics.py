import numpy as np
import pytest

from graphhall.metrics import classification_metrics


def test_classification_metrics_use_hallucination_positive_and_factual_calibration() -> None:
    labels = np.asarray([0, 1, 1, 0])
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]], dtype=np.float64)

    result = classification_metrics(labels, probabilities, bins=2)

    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["auroc"] == 0.75
    assert result["auprc"] == pytest.approx(5 / 6)
    assert result["ece"] == pytest.approx(0.175)
    assert result["brier"] == pytest.approx(0.2125)
    assert result["nll"] == pytest.approx(-np.log([0.8, 0.7, 0.4, 0.4]).mean())


@pytest.mark.parametrize(
    ("labels", "probabilities", "bins", "message"),
    [
        ([0, 1], [[0.8, 0.2]], 10, "shape"),
        ([0], [[0.4, 0.4]], 10, "sum to 1"),
        ([0], [[np.nan, np.nan]], 10, "finite"),
        ([0], [[-0.1, 1.1]], 10, "within"),
        ([2], [[0.5, 0.5]], 10, "binary"),
        ([0], [[0.5, 0.5]], 0, "positive integer"),
    ],
)
def test_classification_metrics_reject_invalid_inputs(labels, probabilities, bins, message) -> None:
    with pytest.raises(ValueError, match=message):
        classification_metrics(labels, probabilities, bins=bins)


def test_rank_metrics_are_null_for_single_class_evaluation() -> None:
    result = classification_metrics([1, 1], [[0.2, 0.8], [0.1, 0.9]])

    assert result["auroc"] is None
    assert result["auprc"] is None
