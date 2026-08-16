"""Deterministic checks for the Day 3 margin and tree implementations."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch

from basic_ml_mini_projects.day01_supervised_basic import (
    generate_binary_classification_data,
)
from basic_ml_mini_projects.day03_margins_trees import (
    decision_function_gradient_boosting,
    decision_function_linear_svm,
    fit_decision_tree,
    fit_lda,
    fit_linear_svm,
    fit_logistic_gradient_boosting,
    fit_random_forest,
    generate_mixed_signal_data,
    make_margin_splits,
    make_mixed_signal_splits,
    predict_decision_tree,
    predict_gradient_boosting,
    predict_gradient_boosting_probabilities,
    predict_lda,
    predict_lda_probabilities,
    predict_lda_scores,
    predict_linear_svm,
    predict_random_forest,
    predict_random_forest_probabilities,
    run_day03_demo,
)


CSV_HEADER = (
    "section,dataset,model,setting,train_accuracy,validation_accuracy,"
    "test_accuracy,train_validation_gap,test_tn,test_fp,test_fn,test_tp,"
    "initial_loss,final_loss,training_steps,feature_split_counts"
)
CSV_FIELDS = tuple(CSV_HEADER.split(","))
FLOAT_CSV_FIELDS = {
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "train_validation_gap",
    "initial_loss",
    "final_loss",
}
INTEGER_CSV_FIELDS = {
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "training_steps",
}


def _split_tensors(splits: object) -> tuple[torch.Tensor, ...]:
    return (
        splits.x_train,
        splits.y_train,
        splits.x_validation,
        splits.y_validation,
        splits.x_test,
        splits.y_test,
        splits.feature_mean,
        splits.feature_std,
    )


def _tree_depth(node: object) -> int:
    if node.feature_index is None:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    return (labels == predictions).to(torch.float32).mean().item()


def _frozen_mixed_signal_data(
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples, 6), generator=generator)
    noise = torch.randn(n_samples, generator=generator)
    interaction = torch.where(x[:, 3] * x[:, 4] >= 0, 1.0, -1.0)
    score = (
        1.50 * x[:, 0]
        + 0.75 * x[:, 1]
        - 0.25 * x[:, 2]
        + 1.25 * interaction
        + 0.35 * noise
    )
    return x, (score >= 0).to(torch.long)


def _standardize_expected_splits(
    raw_train: torch.Tensor,
    raw_validation: torch.Tensor,
    raw_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw_train.mean(dim=0, keepdim=True)
    std = raw_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (
        (raw_train - mean) / std,
        (raw_validation - mean) / std,
        (raw_test - mean) / std,
        mean,
        std,
    )


def _expected_margin_split_tensors(
    n_samples: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    pool_x, raw_test_x, pool_y, test_y = generate_binary_classification_data(
        n_samples=n_samples,
        test_fraction=test_fraction,
        seed=seed,
    )
    permutation = torch.randperm(
        pool_x.shape[0], generator=torch.Generator().manual_seed(seed + 1)
    )
    validation_count = max(1, int(validation_fraction * n_samples))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    raw_train_x = pool_x[train_indices]
    raw_validation_x = pool_x[validation_indices]
    train_x, validation_x, test_x, mean, std = _standardize_expected_splits(
        raw_train_x,
        raw_validation_x,
        raw_test_x,
    )
    return (
        train_x,
        pool_y[train_indices].reshape(-1).to(torch.long),
        validation_x,
        pool_y[validation_indices].reshape(-1).to(torch.long),
        test_x,
        test_y.reshape(-1).to(torch.long),
        mean,
        std,
    )


def _expected_mixed_split_tensors(
    n_samples: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    x, y = _frozen_mixed_signal_data(n_samples, seed)
    permutation = torch.randperm(
        n_samples, generator=torch.Generator().manual_seed(seed + 1)
    )
    train_count = int(train_fraction * n_samples)
    validation_count = int(validation_fraction * n_samples)
    train_indices = permutation[:train_count]
    validation_indices = permutation[train_count : train_count + validation_count]
    test_indices = permutation[train_count + validation_count :]
    train_x, validation_x, test_x, mean, std = _standardize_expected_splits(
        x[train_indices],
        x[validation_indices],
        x[test_indices],
    )
    return (
        train_x,
        y[train_indices],
        validation_x,
        y[validation_indices],
        test_x,
        y[test_indices],
        mean,
        std,
    )


def _assert_detached_finite(
    test_case: unittest.TestCase,
    *tensors: torch.Tensor,
) -> None:
    for tensor in tensors:
        test_case.assertIsInstance(tensor, torch.Tensor)
        test_case.assertFalse(tensor.requires_grad)
        test_case.assertTrue(torch.isfinite(tensor).all().item())


def _assert_frozen(
    test_case: unittest.TestCase,
    instance: object,
    field: str,
    replacement: object,
) -> None:
    with test_case.assertRaises(FrozenInstanceError):
        setattr(instance, field, replacement)


def _serialize_returned_row(
    row: dict[str, str | int | float],
) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for field in CSV_FIELDS:
        value = row[field]
        if value == "":
            serialized[field] = ""
        elif field in FLOAT_CSV_FIELDS:
            serialized[field] = format(float(value), ".10g")
        elif field in INTEGER_CSV_FIELDS:
            serialized[field] = str(int(value))
        else:
            serialized[field] = str(value)
    return serialized


class Day03MarginsTreesTests(unittest.TestCase):
    def test_data_and_splits_are_reproducible_standardized_and_shaped(
        self,
    ) -> None:
        expected_x, expected_y = _frozen_mixed_signal_data(150, seed=7)

        generated_x, generated_y = generate_mixed_signal_data(150, seed=7)
        repeated_x, repeated_y = generate_mixed_signal_data(150, seed=7)
        self.assertTrue(torch.equal(generated_x, expected_x))
        self.assertTrue(torch.equal(generated_y, expected_y))
        self.assertTrue(torch.equal(generated_x, repeated_x))
        self.assertTrue(torch.equal(generated_y, repeated_y))
        _assert_detached_finite(self, generated_x, generated_y)

        cases = (
            (
                make_margin_splits,
                {},
                ((180, 2), (60, 2), (60, 2)),
                _expected_margin_split_tensors(300, 0.20, 0.20, 42),
            ),
            (
                make_margin_splits,
                {
                    "n_samples": 150,
                    "validation_fraction": 0.20,
                    "test_fraction": 0.20,
                    "seed": 7,
                },
                ((90, 2), (30, 2), (30, 2)),
                _expected_margin_split_tensors(150, 0.20, 0.20, 7),
            ),
            (
                make_mixed_signal_splits,
                {},
                ((288, 6), (96, 6), (96, 6)),
                _expected_mixed_split_tensors(480, 0.60, 0.20, 42),
            ),
            (
                make_mixed_signal_splits,
                {
                    "n_samples": 150,
                    "train_fraction": 0.60,
                    "validation_fraction": 0.20,
                    "seed": 7,
                },
                ((90, 6), (30, 6), (30, 6)),
                _expected_mixed_split_tensors(150, 0.60, 0.20, 7),
            ),
        )

        for split_function, arguments, expected_shapes, expected_tensors in cases:
            with self.subTest(split_function=split_function.__name__, arguments=arguments):
                first = split_function(**arguments)
                second = split_function(**arguments)
                actual_tensors = _split_tensors(first)

                self.assertTrue(
                    all(
                        torch.equal(left, right)
                        for left, right in zip(
                            _split_tensors(first), _split_tensors(second)
                        )
                    )
                )
                for actual, expected in zip(actual_tensors, expected_tensors):
                    if actual.dtype.is_floating_point:
                        self.assertTrue(
                            torch.allclose(actual, expected, atol=1e-7, rtol=0.0)
                        )
                    else:
                        self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(first.x_train.shape, expected_shapes[0])
                self.assertEqual(first.x_validation.shape, expected_shapes[1])
                self.assertEqual(first.x_test.shape, expected_shapes[2])
                for features in (
                    first.x_train,
                    first.x_validation,
                    first.x_test,
                ):
                    self.assertTrue(torch.isfinite(features).all().item())
                self.assertTrue(torch.isfinite(first.feature_mean).all().item())
                self.assertTrue(torch.isfinite(first.feature_std).all().item())
                self.assertTrue((first.feature_std > 0.0).all().item())
                for labels, expected_count in zip(
                    (first.y_train, first.y_validation, first.y_test),
                    (shape[0] for shape in expected_shapes),
                ):
                    self.assertEqual(labels.shape, (expected_count,))
                    self.assertEqual(labels.dtype, torch.long)
                    self.assertEqual(torch.unique(labels).tolist(), [0, 1])
                self.assertTrue(
                    torch.allclose(
                        first.x_train.mean(dim=0),
                        torch.zeros(first.x_train.shape[1]),
                        atol=1e-6,
                    )
                )
                _assert_detached_finite(self, *actual_tensors)
                _assert_frozen(self, first, "x_train", first.x_train.clone())
                self.assertTrue(
                    torch.allclose(
                        first.x_train.std(dim=0, unbiased=False),
                        torch.ones(first.x_train.shape[1]),
                        atol=1e-6,
                    )
                )

    def test_linear_svm_fits_frozen_separable_fixture_deterministically(
        self,
    ) -> None:
        x = torch.tensor(
            [[-2.0, -1.0], [-1.0, -2.0], [-2.0, -2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        arguments = {
            "epochs": 200,
            "learning_rate": 0.05,
            "batch_size": 6,
            "l2_penalty": 0.01,
            "seed": 7,
        }

        first = fit_linear_svm(x, y, **arguments)
        second = fit_linear_svm(x, y, **arguments)
        scores = decision_function_linear_svm(x, first)
        predictions = predict_linear_svm(x, first)

        self.assertTrue(torch.equal(predictions, y))
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(torch.isfinite(scores).all().item())
        self.assertEqual(len(first.loss_history), 201)
        self.assertLess(first.loss_history[-1], 0.25 * first.loss_history[0])
        self.assertTrue(torch.equal(first.weights, second.weights))
        self.assertTrue(torch.equal(first.bias, second.bias))
        self.assertEqual(first.loss_history, second.loss_history)
        self.assertTrue(all(math.isfinite(loss) for loss in first.loss_history))
        _assert_detached_finite(
            self,
            first.weights,
            first.bias,
            scores,
            predictions,
        )
        _assert_frozen(self, first, "weights", first.weights.clone())

    def test_decision_tree_uses_frozen_tie_break_and_threshold(self) -> None:
        x = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]])
        y = torch.tensor([0, 0, 1, 1])

        model = fit_decision_tree(
            x,
            y,
            max_depth=1,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            seed=3,
        )

        predictions = predict_decision_tree(x, model)

        self.assertEqual(model.root.feature_index, 0)
        self.assertAlmostEqual(model.root.threshold, 0.5)
        self.assertTrue(math.isfinite(model.root.threshold))
        self.assertEqual(model.split_counts, (1, 0))
        self.assertTrue(torch.equal(predictions, y))
        self.assertIsNotNone(model.root.left)
        self.assertIsNotNone(model.root.right)
        for leaf, expected_prediction in ((model.root.left, 0), (model.root.right, 1)):
            self.assertEqual(leaf.prediction, expected_prediction)
            self.assertIsNone(leaf.feature_index)
            self.assertIsNone(leaf.threshold)
            self.assertIsNone(leaf.left)
            self.assertIsNone(leaf.right)

        tied_leaf = fit_decision_tree(
            x,
            y,
            max_depth=0,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        self.assertEqual(tied_leaf.root.prediction, 0)

        adjacent_lower = torch.tensor(1.0000001192092896, dtype=torch.float32)
        adjacent_upper = torch.nextafter(
            adjacent_lower,
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        adjacent_x = torch.stack((adjacent_lower, adjacent_upper)).reshape(-1, 1)
        adjacent_y = torch.tensor([0, 1])
        adjacent_model = fit_decision_tree(
            adjacent_x,
            adjacent_y,
            max_depth=1,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        self.assertGreaterEqual(adjacent_model.root.threshold, adjacent_lower.item())
        self.assertLess(adjacent_model.root.threshold, adjacent_upper.item())
        self.assertTrue(
            torch.equal(
                predict_decision_tree(adjacent_x, adjacent_model),
                adjacent_y,
            )
        )
        _assert_detached_finite(self, predictions)
        _assert_frozen(self, model, "max_depth", 0)
        _assert_frozen(self, model.root, "prediction", 1)

    def test_decision_tree_leaf_stops_and_mixed_signal_accuracy(self) -> None:
        constant_x = torch.zeros((4, 2))
        majority_y = torch.tensor([0, 1, 1, 1])

        constant_model = fit_decision_tree(
            constant_x,
            majority_y,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        depth_zero_model = fit_decision_tree(
            constant_x,
            majority_y,
            max_depth=0,
            min_samples_split=2,
            min_samples_leaf=1,
        )

        for model in (constant_model, depth_zero_model):
            self.assertEqual(model.root.prediction, 1)
            self.assertEqual(model.root.class_counts, (1, 3))
            self.assertIsNone(model.root.feature_index)
            self.assertEqual(model.split_counts, (0, 0))

        splits = make_mixed_signal_splits(
            n_samples=240,
            train_fraction=0.60,
            validation_fraction=0.20,
            seed=11,
        )
        model = fit_decision_tree(
            splits.x_train,
            splits.y_train,
            max_depth=3,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=None,
            seed=11,
        )
        predictions = predict_decision_tree(splits.x_validation, model)

        self.assertEqual(model.max_depth, 3)
        self.assertLessEqual(_tree_depth(model.root), 3)
        self.assertGreaterEqual(_accuracy(splits.y_validation, predictions), 0.70)
        _assert_detached_finite(self, predictions)

    def test_random_forest_bootstraps_depth_determinism_and_accuracy(self) -> None:
        splits = make_mixed_signal_splits(
            n_samples=240,
            train_fraction=0.60,
            validation_fraction=0.20,
            seed=13,
        )
        arguments = {
            "n_trees": 9,
            "max_depth": 3,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": 2,
            "seed": 17,
        }

        first = fit_random_forest(splits.x_train, splits.y_train, **arguments)
        second = fit_random_forest(splits.x_train, splits.y_train, **arguments)
        probabilities = predict_random_forest_probabilities(
            splits.x_validation, first
        )
        predictions = predict_random_forest(splits.x_validation, first)

        self.assertEqual(len(first.trees), 9)
        self.assertEqual(len(first.bootstrap_indices), 9)
        for tree_index, (bootstrap, repeated_bootstrap) in enumerate(
            zip(first.bootstrap_indices, second.bootstrap_indices)
        ):
            generator = torch.Generator().manual_seed(17 + 1000 + tree_index)
            expected = torch.randint(
                0,
                splits.x_train.shape[0],
                (splits.x_train.shape[0],),
                generator=generator,
            )
            self.assertEqual(bootstrap.dtype, torch.long)
            self.assertEqual(bootstrap.shape, (splits.x_train.shape[0],))
            self.assertTrue(torch.equal(bootstrap, expected))
            self.assertTrue(torch.equal(bootstrap, repeated_bootstrap))
            feature_generator = torch.Generator().manual_seed(17 + tree_index + 1)
            root_candidates = torch.randperm(
                splits.x_train.shape[1], generator=feature_generator
            )[:2].sort().values.tolist()
            self.assertIsNotNone(first.trees[tree_index].root.feature_index)
            self.assertIn(
                first.trees[tree_index].root.feature_index,
                root_candidates,
            )
        self.assertTrue(
            any(
                not torch.equal(first.bootstrap_indices[0], bootstrap)
                for bootstrap in first.bootstrap_indices[1:]
            )
        )
        self.assertTrue(all(_tree_depth(tree.root) <= 3 for tree in first.trees))
        expected_split_counts = tuple(
            sum(tree.split_counts[index] for tree in first.trees)
            for index in range(splits.x_train.shape[1])
        )
        self.assertEqual(first.split_counts, expected_split_counts)
        self.assertEqual(first.split_counts, second.split_counts)
        self.assertTrue(
            torch.equal(
                predictions,
                predict_random_forest(splits.x_validation, second),
            )
        )
        self.assertEqual(probabilities.shape, (splits.x_validation.shape[0],))
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all().item())
        self.assertGreaterEqual(_accuracy(splits.y_validation, predictions), 0.70)
        _assert_detached_finite(
            self,
            *first.bootstrap_indices,
            probabilities,
            predictions,
        )
        _assert_frozen(self, first, "trees", ())

    def test_gradient_boosting_fits_frozen_one_stump_fixture(self) -> None:
        x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        y = torch.tensor([0, 0, 1, 1])

        model = fit_logistic_gradient_boosting(
            x,
            y,
            n_estimators=1,
            learning_rate=0.20,
            min_samples_leaf=1,
        )
        stump = model.stumps[0]

        self.assertEqual(len(model.stumps), 1)
        self.assertEqual(len(model.loss_history), 2)
        self.assertEqual(model.split_counts, (1,))
        self.assertAlmostEqual(model.base_logit, 0.0)
        self.assertEqual(stump.feature_index, 0)
        self.assertAlmostEqual(stump.threshold, 0.0)
        self.assertAlmostEqual(stump.left_value, -0.5)
        self.assertAlmostEqual(stump.right_value, 0.5)
        self.assertAlmostEqual(stump.step_size, 0.20)
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        predictions = predict_gradient_boosting(x, model)
        self.assertTrue(torch.equal(predictions, y))
        self.assertTrue(all(math.isfinite(loss) for loss in model.loss_history))
        self.assertTrue(
            all(
                math.isfinite(value)
                for value in (
                    model.base_logit,
                    stump.threshold,
                    stump.left_value,
                    stump.right_value,
                    stump.step_size,
                )
            )
        )

        adjacent_lower = torch.tensor(1.0000001192092896, dtype=torch.float32)
        adjacent_upper = torch.nextafter(
            adjacent_lower,
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        adjacent_x = torch.stack((adjacent_lower, adjacent_upper)).reshape(-1, 1)
        adjacent_y = torch.tensor([0, 1])
        adjacent_model = fit_logistic_gradient_boosting(
            adjacent_x,
            adjacent_y,
            n_estimators=1,
            learning_rate=0.20,
            min_samples_leaf=1,
        )
        self.assertEqual(len(adjacent_model.stumps), 1)
        self.assertGreaterEqual(
            adjacent_model.stumps[0].threshold,
            adjacent_lower.item(),
        )
        self.assertLess(
            adjacent_model.stumps[0].threshold,
            adjacent_upper.item(),
        )
        self.assertTrue(
            torch.equal(
                predict_gradient_boosting(adjacent_x, adjacent_model),
                adjacent_y,
            )
        )
        _assert_detached_finite(self, predictions)
        _assert_frozen(self, model, "stumps", ())
        _assert_frozen(self, stump, "step_size", 0.0)

    def test_gradient_boosting_outer_band_loss_probabilities_and_accuracy(
        self,
    ) -> None:
        x = torch.tensor(
            [[-3.0], [-2.5], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [2.5], [3.0]]
        )
        y = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

        model = fit_logistic_gradient_boosting(
            x,
            y,
            n_estimators=30,
            learning_rate=0.20,
            min_samples_leaf=1,
        )
        scores = decision_function_gradient_boosting(x, model)
        probabilities = predict_gradient_boosting_probabilities(x, model)
        predictions = predict_gradient_boosting(x, model)

        self.assertEqual(len(model.stumps), 30)
        self.assertEqual(len(model.loss_history), 31)
        self.assertEqual(sum(model.split_counts), 30)
        self.assertTrue(
            all(
                later <= earlier + 1e-6
                for earlier, later in zip(
                    model.loss_history, model.loss_history[1:]
                )
            )
        )
        self.assertLess(model.loss_history[-1], 0.75 * model.loss_history[0])
        self.assertTrue(torch.isfinite(scores).all().item())
        self.assertTrue(torch.isfinite(probabilities).all().item())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all().item())
        self.assertTrue(torch.allclose(probabilities, torch.sigmoid(scores), atol=1e-7))
        self.assertGreaterEqual(_accuracy(y, predictions), 0.90)
        self.assertTrue(all(math.isfinite(loss) for loss in model.loss_history))
        _assert_detached_finite(self, scores, probabilities, predictions)

    def test_lda_matches_frozen_parameters_probabilities_and_labels(self) -> None:
        x = torch.tensor(
            [[-2.0, -1.0], [-2.0, 0.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1])

        model = fit_lda(x, y)
        scores = predict_lda_scores(x, model)
        probabilities = predict_lda_probabilities(x, model)
        predictions = predict_lda(x, model)

        self.assertTrue(torch.equal(model.classes, torch.tensor([0, 1])))
        self.assertTrue(
            torch.allclose(model.class_priors, torch.tensor([0.5, 0.5]), atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(
                model.class_means,
                torch.tensor([[-5.0 / 3.0, -2.0 / 3.0], [5.0 / 3.0, 4.0 / 3.0]]),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(model.covariance).all().item())
        self.assertTrue(
            torch.allclose(model.covariance, model.covariance.T, atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(
                model.cholesky_factor @ model.cholesky_factor.T,
                model.covariance,
                atol=1e-5,
            )
        )
        self.assertAlmostEqual(model.regularization, 1e-4)
        self.assertEqual(scores.shape, (6, 2))
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(6), atol=1e-6)
        )
        self.assertTrue(torch.equal(predictions, y))
        _assert_detached_finite(
            self,
            model.classes,
            model.class_priors,
            model.class_means,
            model.covariance,
            model.cholesky_factor,
            scores,
            probabilities,
            predictions,
        )
        _assert_frozen(self, model, "regularization", 1.0)

    def test_invalid_data_models_hyperparameters_labels_and_widths_are_rejected(
        self,
    ) -> None:
        base_x = torch.tensor(
            [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
        )
        base_y = torch.tensor([0, 0, 1, 1])

        fitters = (
            lambda x, y: fit_linear_svm(x, y, epochs=1, batch_size=4),
            lambda x, y: fit_decision_tree(
                x,
                y,
                max_depth=1,
                min_samples_split=2,
                min_samples_leaf=1,
            ),
            lambda x, y: fit_random_forest(
                x,
                y,
                n_trees=1,
                max_depth=1,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1,
            ),
            lambda x, y: fit_logistic_gradient_boosting(
                x, y, n_estimators=1, min_samples_leaf=1
            ),
            lambda x, y: fit_lda(x, y),
        )
        common_bad_inputs = (
            ("not a tensor", base_y),
            (base_x, "not a tensor"),
            (torch.ones(4), base_y),
            (torch.empty((0, 2)), torch.empty(0, dtype=torch.long)),
            (torch.empty((4, 0)), base_y),
            (
                torch.tensor(
                    [[float("nan"), 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
                ),
                base_y,
            ),
            (base_x, base_y[:3]),
            (base_x, base_y.unsqueeze(1)),
            (base_x, torch.tensor([0, 0, 1, 2])),
            (base_x, torch.tensor([0.0, 0.0, 1.0, float("nan")])),
        )
        for fitter in fitters:
            for bad_x, bad_y in common_bad_inputs:
                with self.subTest(fitter=fitter, bad_x=bad_x, bad_y=bad_y):
                    with self.assertRaises((TypeError, ValueError)):
                        fitter(bad_x, bad_y)

        rejection_calls = (
            lambda: generate_mixed_signal_data(0),
            lambda: generate_mixed_signal_data(True),
            lambda: generate_mixed_signal_data(10, seed=True),
            lambda: make_margin_splits(validation_fraction=0.0),
            lambda: make_margin_splits(validation_fraction=-0.1),
            lambda: make_margin_splits(validation_fraction=1.0),
            lambda: make_margin_splits(validation_fraction=True),
            lambda: make_margin_splits(validation_fraction=float("nan")),
            lambda: make_margin_splits(test_fraction=0.0),
            lambda: make_margin_splits(test_fraction=-0.1),
            lambda: make_margin_splits(test_fraction=1.0),
            lambda: make_margin_splits(test_fraction=True),
            lambda: make_margin_splits(test_fraction=float("inf")),
            lambda: make_margin_splits(validation_fraction=0.5, test_fraction=0.5),
            lambda: make_margin_splits(9, 0.49, 0.49, seed=7),
            lambda: make_margin_splits(seed=True),
            lambda: make_mixed_signal_splits(train_fraction=0.0),
            lambda: make_mixed_signal_splits(train_fraction=-0.1),
            lambda: make_mixed_signal_splits(train_fraction=1.0),
            lambda: make_mixed_signal_splits(train_fraction=True),
            lambda: make_mixed_signal_splits(train_fraction=float("nan")),
            lambda: make_mixed_signal_splits(validation_fraction=0.0),
            lambda: make_mixed_signal_splits(validation_fraction=-0.1),
            lambda: make_mixed_signal_splits(validation_fraction=1.0),
            lambda: make_mixed_signal_splits(validation_fraction=True),
            lambda: make_mixed_signal_splits(validation_fraction=float("inf")),
            lambda: make_mixed_signal_splits(train_fraction=0.8, validation_fraction=0.2),
            lambda: make_mixed_signal_splits(2, 0.40, 0.40, seed=7),
            lambda: make_mixed_signal_splits(10, 0.90, 0.05, seed=7),
            lambda: make_mixed_signal_splits(seed=True),
            lambda: fit_linear_svm(base_x, base_y, epochs=0),
            lambda: fit_linear_svm(base_x, base_y, epochs=True),
            lambda: fit_linear_svm(base_x, base_y, learning_rate=0.0),
            lambda: fit_linear_svm(base_x, base_y, learning_rate=True),
            lambda: fit_linear_svm(base_x, base_y, batch_size=0),
            lambda: fit_linear_svm(base_x, base_y, batch_size=True),
            lambda: fit_linear_svm(base_x, base_y, l2_penalty=-1.0),
            lambda: fit_linear_svm(base_x, base_y, l2_penalty=True),
            lambda: fit_linear_svm(base_x, base_y, seed=True),
            lambda: fit_linear_svm(base_x, torch.zeros(4, dtype=torch.long)),
            lambda: fit_decision_tree(base_x, base_y, max_depth=-1),
            lambda: fit_decision_tree(base_x, base_y, max_depth=True),
            lambda: fit_decision_tree(base_x, base_y, min_samples_split=1),
            lambda: fit_decision_tree(base_x, base_y, min_samples_split=True),
            lambda: fit_decision_tree(base_x, base_y, min_samples_leaf=0),
            lambda: fit_decision_tree(base_x, base_y, min_samples_leaf=True),
            lambda: fit_decision_tree(base_x, base_y, max_features=0),
            lambda: fit_decision_tree(base_x, base_y, max_features=3),
            lambda: fit_decision_tree(base_x, base_y, max_features=True),
            lambda: fit_decision_tree(base_x, base_y, seed=True),
            lambda: fit_random_forest(base_x, base_y, n_trees=0),
            lambda: fit_random_forest(base_x, base_y, n_trees=True),
            lambda: fit_random_forest(base_x, base_y, max_depth=-1),
            lambda: fit_random_forest(base_x, base_y, max_depth=True),
            lambda: fit_random_forest(base_x, base_y, min_samples_split=1),
            lambda: fit_random_forest(base_x, base_y, min_samples_split=True),
            lambda: fit_random_forest(base_x, base_y, min_samples_leaf=0),
            lambda: fit_random_forest(base_x, base_y, min_samples_leaf=True),
            lambda: fit_random_forest(base_x, base_y, max_features=0),
            lambda: fit_random_forest(base_x, base_y, max_features=3),
            lambda: fit_random_forest(base_x, base_y, max_features=True),
            lambda: fit_random_forest(base_x, base_y, seed=True),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, n_estimators=0),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, n_estimators=True),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, learning_rate=0.0),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, learning_rate=True),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, min_samples_leaf=0),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, min_samples_leaf=True),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, loss_tolerance=-1.0),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, loss_tolerance=True),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, max_backtracking_steps=-1),
            lambda: fit_logistic_gradient_boosting(base_x, base_y, max_backtracking_steps=True),
            lambda: fit_logistic_gradient_boosting(
                base_x, torch.zeros(4, dtype=torch.long)
            ),
            lambda: fit_lda(base_x, base_y, covariance_regularization=0.0),
            lambda: fit_lda(base_x, base_y, covariance_regularization=True),
            lambda: fit_lda(base_x, base_y, max_cholesky_attempts=0),
            lambda: fit_lda(base_x, base_y, max_cholesky_attempts=True),
            lambda: fit_lda(base_x, torch.zeros(4, dtype=torch.long)),
            lambda: fit_lda(base_x, torch.tensor([0, 0, 0, 1])),
        )
        for rejection_call in rejection_calls:
            with self.subTest(call=rejection_call):
                with self.assertRaises((TypeError, ValueError)):
                    rejection_call()

        svm = fit_linear_svm(base_x, base_y, epochs=2, batch_size=4)
        tree = fit_decision_tree(
            base_x,
            base_y,
            max_depth=1,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        forest = fit_random_forest(
            base_x,
            base_y,
            n_trees=1,
            max_depth=1,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1,
        )
        boosting = fit_logistic_gradient_boosting(
            base_x, base_y, n_estimators=1, min_samples_leaf=1
        )
        lda = fit_lda(base_x, base_y)
        wrong_width = torch.zeros((2, 3))
        width_calls = (
            lambda: decision_function_linear_svm(wrong_width, svm),
            lambda: predict_linear_svm(wrong_width, svm),
            lambda: predict_decision_tree(wrong_width, tree),
            lambda: predict_random_forest_probabilities(wrong_width, forest),
            lambda: predict_random_forest(wrong_width, forest),
            lambda: decision_function_gradient_boosting(wrong_width, boosting),
            lambda: predict_gradient_boosting_probabilities(wrong_width, boosting),
            lambda: predict_gradient_boosting(wrong_width, boosting),
            lambda: predict_lda_scores(wrong_width, lda),
            lambda: predict_lda_probabilities(wrong_width, lda),
            lambda: predict_lda(wrong_width, lda),
        )
        for width_call in width_calls:
            with self.subTest(call=width_call):
                with self.assertRaises((TypeError, ValueError)):
                    width_call()

        invalid_prediction_data_calls = (
            lambda: predict_linear_svm(torch.ones(2), svm),
            lambda: predict_decision_tree(
                torch.tensor([[float("inf"), 0.0]]), tree
            ),
            lambda: predict_random_forest(torch.empty((2, 0)), forest),
            lambda: predict_gradient_boosting("not a tensor", boosting),
            lambda: predict_lda(torch.empty((0, 2)), lda),
        )
        for invalid_prediction_call in invalid_prediction_data_calls:
            with self.subTest(call=invalid_prediction_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_prediction_call()

        invalid_model_calls = (
            lambda: decision_function_linear_svm(base_x, None),
            lambda: predict_linear_svm(base_x, None),
            lambda: predict_decision_tree(base_x, None),
            lambda: predict_random_forest_probabilities(base_x, None),
            lambda: predict_random_forest(base_x, None),
            lambda: decision_function_gradient_boosting(base_x, None),
            lambda: predict_gradient_boosting_probabilities(base_x, None),
            lambda: predict_gradient_boosting(base_x, None),
            lambda: predict_lda_scores(base_x, None),
            lambda: predict_lda_probabilities(base_x, None),
            lambda: predict_lda(base_x, None),
        )
        for invalid_model_call in invalid_model_calls:
            with self.subTest(call=invalid_model_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_model_call()

    def test_demo_writes_exact_ordered_deterministic_csv(self) -> None:
        expected_rows = (
            (
                "margin_check",
                "day01_separable_blobs",
                "linear SVM",
                "epochs=300;lr=0.05;l2=0.01",
            ),
            (
                "tabular_comparison",
                "mixed_signal_tabular",
                "linear SVM",
                "epochs=300;lr=0.05;l2=0.01",
            ),
            (
                "tabular_comparison",
                "mixed_signal_tabular",
                "decision tree",
                "depth=3;min_split=12;min_leaf=6",
            ),
            (
                "tabular_comparison",
                "mixed_signal_tabular",
                "random forest",
                "trees=25;depth=4;max_features=2",
            ),
            (
                "tabular_comparison",
                "mixed_signal_tabular",
                "logistic gradient boosting",
                "estimators=50;lr=0.20;min_leaf=8",
            ),
            (
                "tabular_comparison",
                "mixed_signal_tabular",
                "LDA",
                "regularization=0.0001",
            ),
        )
        accuracy_floors = (0.95, 0.70, 0.72, 0.72, 0.70, 0.68)

        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_output = Path(first_directory)
            second_output = Path(second_directory)
            first_returned_rows = run_day03_demo(first_output)
            second_returned_rows = run_day03_demo(second_output)
            first_csv = first_output / "day03_metrics.csv"
            second_csv = second_output / "day03_metrics.csv"

            self.assertEqual(len(first_returned_rows), 6)
            self.assertEqual(len(second_returned_rows), 6)
            self.assertEqual(
                first_csv.read_text(encoding="utf-8").splitlines()[0],
                CSV_HEADER,
            )
            with first_csv.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, CSV_HEADER.split(","))
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                [_serialize_returned_row(row) for row in first_returned_rows],
                rows,
            )
            self.assertEqual(
                [_serialize_returned_row(row) for row in second_returned_rows],
                rows,
            )
            self.assertTrue(
                all(set(row) == set(CSV_FIELDS) for row in first_returned_rows)
            )
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())
            self.assertEqual(
                {path.name for path in first_output.iterdir()},
                {"day03_metrics.csv"},
            )
            self.assertEqual(
                {path.name for path in second_output.iterdir()},
                {"day03_metrics.csv"},
            )

            for index, (row, expected, accuracy_floor) in enumerate(
                zip(rows, expected_rows, accuracy_floors)
            ):
                self.assertEqual(
                    (
                        row["section"],
                        row["dataset"],
                        row["model"],
                        row["setting"],
                    ),
                    expected,
                )
                self.assertGreaterEqual(float(row["test_accuracy"]), accuracy_floor)
                expected_test_count = 60 if index == 0 else 96
                confusion_total = sum(
                    int(row[field])
                    for field in ("test_tn", "test_fp", "test_fn", "test_tp")
                )
                self.assertEqual(confusion_total, expected_test_count)
                self.assertAlmostEqual(
                    float(row["train_validation_gap"]),
                    float(row["train_accuracy"])
                    - float(row["validation_accuracy"]),
                    places=8,
                )
                for field in FLOAT_CSV_FIELDS:
                    if row[field] != "":
                        self.assertEqual(
                            row[field],
                            format(float(row[field]), ".10g"),
                        )
                for field in INTEGER_CSV_FIELDS:
                    if row[field] != "":
                        self.assertEqual(row[field], str(int(row[field])))

            for index in (0, 1):
                self.assertNotEqual(rows[index]["initial_loss"], "")
                self.assertNotEqual(rows[index]["final_loss"], "")
                self.assertEqual(rows[index]["training_steps"], "300")
            self.assertNotEqual(rows[4]["initial_loss"], "")
            self.assertNotEqual(rows[4]["final_loss"], "")
            self.assertGreaterEqual(int(rows[4]["training_steps"]), 1)
            self.assertLessEqual(int(rows[4]["training_steps"]), 50)
            for index in (2, 3, 5):
                self.assertEqual(rows[index]["initial_loss"], "")
                self.assertEqual(rows[index]["final_loss"], "")
                self.assertEqual(rows[index]["training_steps"], "")
            for index in (2, 3, 4):
                counts = rows[index]["feature_split_counts"].split("|")
                self.assertEqual(len(counts), 6)
                self.assertTrue(all(int(count) >= 0 for count in counts))
                self.assertEqual(
                    rows[index]["feature_split_counts"],
                    "|".join(str(int(count)) for count in counts),
                )
            for index in (0, 1, 5):
                self.assertEqual(rows[index]["feature_split_counts"], "")


if __name__ == "__main__":
    unittest.main()
