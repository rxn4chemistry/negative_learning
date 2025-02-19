import numpy as np
import pytest
import torch
from rxn_negative_learning.models.baseline.baseline_model import Baseline, BaselineTrainer
from rxn_negative_learning.utils.repo_utils import reset_random_seed


@pytest.fixture
def data():
    reset_random_seed()

    n_samples, n_features = 10, 20
    rng = np.random.RandomState(0)
    Y = torch.tensor(rng.rand(n_samples)).float()
    X = torch.tensor(rng.randn(n_samples, n_features)).float()
    return X, Y, n_samples, n_features


def test_baseline(data):
    X, Y, n_samples, n_features = data
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    for i in range(0, 5):
        baseline_trainer.train_step(X, Y)
    print(baseline_trainer.loss_history)
    assert baseline_trainer.loss_history == pytest.approx([
        0.13040247559547424,
        45.01543426513672,
        1.8061559200286865,
        0.3720577657222748,
        1.4421032667160034,
    ])

    Y_pred = baseline_trainer.predict(X)
    assert float(torch.mean(baseline_trainer.loss(Y_pred, Y))) == pytest.approx(2.244994878768921)


def test_baseline_with_batch(data):
    X, Y, n_samples, n_features = data
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    baseline_trainer.train_step(X, Y, batch_size=2)
    print(baseline_trainer.loss_history)
    assert baseline_trainer.loss_history == pytest.approx([8.812806714858327])

    Y_pred = baseline_trainer.predict(X)
    assert float(torch.mean(baseline_trainer.loss(Y_pred, Y))) == pytest.approx(7.152345657348633)


def test_baseline_with_mask(data):
    X, Y, n_samples, n_features = data
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    mask = torch.randint(0, 2, (n_samples,))
    for i in range(0, 5):
        baseline_trainer.train_step(X, Y, mask)
    print(baseline_trainer.loss_history)
    assert baseline_trainer.loss_history == pytest.approx([
        0.14094801247119904,
        40.735084533691406,
        1.3518527746200562,
        1.451961636543274,
        1.6346632242202759,
    ])

    Y_pred = baseline_trainer.predict(X)
    assert float(torch.mean(baseline_trainer.loss(Y_pred, Y))) == pytest.approx(6.204676628112793)


def test_baseline_with_preprocess(data):
    X, Y, n_samples, n_features = data
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    Xp = baseline_trainer.preprocess_step(X)
    for i in range(0, 5):
        baseline_trainer.train_step(Xp, Y)
    print(baseline_trainer.loss_history)
    assert baseline_trainer.loss_history == pytest.approx([
        0.16866159439086914,
        3.746967315673828,
        0.05012155696749687,
        0.20075677335262299,
        0.3754490911960602,
    ])

    Y_pred = baseline_trainer.predict(Xp)
    assert float(torch.mean(baseline_trainer.loss(Y_pred, Y))) == pytest.approx(0.46640530228614807)

    # Full step: preprocess and training combined
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    for i in range(0, 5):
        baseline_trainer.full_step(X, Y)
    print(baseline_trainer.loss_history)
    assert baseline_trainer.loss_history == pytest.approx([
        0.16866159439086914,
        3.746967315673828,
        0.05012155696749687,
        0.20075677335262299,
        0.3754490911960602,
    ])

    Y_pred = baseline_trainer.predict(baseline_trainer.preprocess_step(X))
    assert float(torch.mean(baseline_trainer.loss(Y_pred, Y))) == pytest.approx(0.46640530228614807)


def test_baseline_randomic(data):
    X, Y, n_samples, n_features = data
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    Xp = baseline_trainer.preprocess_step(X)
    for i in range(0, 5):
        baseline_trainer.train_step(Xp, Y)
    print(baseline_trainer.loss_history)
    assert baseline_trainer.loss_history == pytest.approx([
        0.16866159439086914,
        3.746967315673828,
        0.05012155696749687,
        0.20075677335262299,
        0.3754490911960602,
    ])

    Y_pred = baseline_trainer.predict(Xp, randomic=True)
    assert float(torch.mean(baseline_trainer.loss(Y_pred, Y))) == pytest.approx(0.1121731847524643)


def test_oversample_minority_dataset(data):
    X, Y, n_samples, n_features = data
    torch.manual_seed(42)
    baseline = Baseline(n_features)
    baseline_trainer = BaselineTrainer(baseline, learning_rate=0.1, weight_decay=0.0)
    output = baseline_trainer.oversample_minority_dataset(X, Y)

    new_Y = sorted([
        0.5448831915855408,
        0.7151893377304077,
        0.6027633547782898,
        0.54881352186203,
        0.6458941102027893,
        0.891772985458374,
        0.9636627435684204,
        0.42365479469299316,
        0.42365479469299316,
        0.3834415078163147,
        0.4375872015953064,
        0.4375872015953064,
        0.3834415078163147,
        0.3834415078163147,
    ])
    assert sorted(output[1].tolist()) == pytest.approx(new_Y)

    n_positives = len([elem for elem in output[1] if elem > 0.5])
    n_negatives = len([elem for elem in output[1] if elem <= 0.5])
    assert n_positives == n_negatives
