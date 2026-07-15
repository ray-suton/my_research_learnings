"""Deterministic checks for the Day 1 supervised-learning implementations."""

import unittest

import torch

from basic_ml_mini_projects.day01_supervised_basic import (
    classification_metrics,
    confusion_matrix,
    fit_gaussian_naive_bayes,
    fit_linear_regression_closed_form,
    fit_ridge_regression,
    generate_binary_classification_data,
    generate_data,
    predict_gaussian_nb_labels,
    predict_gaussian_nb_probabilities,
    predict_knn,
    predict_logistic_labels,
    predict_logistic_probabilities,
    standardize_from_train,
    train_linear_reg,
    train_logistic_regression,
)


class Day01SupervisedBasicsTests(unittest.TestCase):
    def test_regression_data_is_reproducible_and_has_expected_shapes(self) -> None:
        first = generate_data(n_samples=50, dimensions=2, seed=7)
        second = generate_data(n_samples=50, dimensions=2, seed=7)

        self.assertEqual(first[0].shape, (40, 2))
        self.assertEqual(first[1].shape, (10, 2))
        self.assertEqual(first[2].shape, (40, 1))
        self.assertEqual(first[3].shape, (10, 1))
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(first, second))
        )

    def test_linear_gradient_descent_matches_closed_form(self) -> None:
        x_train, _, y_train, _ = generate_data(
            n_samples=140,
            noise_std=0.1,
            dimensions=2,
            seed=11,
        )
        gd_weights, gd_bias, losses = train_linear_reg(
            x_train,
            y_train,
            epochs=1_000,
            learning_rate=0.03,
        )
        closed_weights, closed_bias = fit_linear_regression_closed_form(
            x_train, y_train
        )

        self.assertLess(losses[-1], losses[0])
        self.assertTrue(torch.allclose(gd_weights, closed_weights, atol=0.02))
        self.assertTrue(torch.allclose(gd_bias, closed_bias, atol=0.02))

    def test_ridge_zero_matches_ols_and_positive_penalty_shrinks_weights(
        self,
    ) -> None:
        x_train, _, y_train, _ = generate_data(
            n_samples=120,
            dimensions=2,
            seed=19,
        )
        ols_weights, ols_bias = fit_linear_regression_closed_form(
            x_train, y_train
        )
        zero_weights, zero_bias = fit_ridge_regression(
            x_train, y_train, l2_penalty=0.0
        )
        ridge_weights, _ = fit_ridge_regression(
            x_train, y_train, l2_penalty=20.0
        )
        heavily_regularized_weights, heavily_regularized_bias = (
            fit_ridge_regression(x_train, y_train, l2_penalty=1_000_000.0)
        )

        self.assertTrue(torch.allclose(zero_weights, ols_weights, atol=1e-6))
        self.assertTrue(torch.allclose(zero_bias, ols_bias, atol=1e-6))
        self.assertLess(ridge_weights.norm(), ols_weights.norm())
        self.assertLess(heavily_regularized_weights.norm(), ridge_weights.norm())
        self.assertAlmostEqual(
            heavily_regularized_bias.item(), y_train.mean().item(), places=3
        )
        with self.assertRaises(ValueError):
            fit_ridge_regression(x_train, y_train, l2_penalty=-0.1)

    def test_logistic_regression_reduces_loss_and_classifies_held_out_data(
        self,
    ) -> None:
        x_train, x_test, y_train, y_test = generate_binary_classification_data(
            n_samples=240,
            seed=23,
        )
        x_train, x_test, _, _ = standardize_from_train(x_train, x_test)
        weights, bias, losses = train_logistic_regression(
            x_train,
            y_train,
            epochs=500,
            learning_rate=0.1,
        )
        probabilities = predict_logistic_probabilities(x_test, weights, bias)
        labels = predict_logistic_labels(x_test, weights, bias)
        metrics = classification_metrics(y_test, labels, probabilities)

        self.assertLess(losses[-1], losses[0])
        self.assertTrue(torch.allclose(x_train.mean(dim=0), torch.zeros(2), atol=1e-6))
        self.assertEqual(y_train.shape, (180, 1))
        self.assertEqual(y_test.shape, (60, 1))
        self.assertTrue(
            torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
        )
        self.assertLessEqual(set(labels.reshape(-1).tolist()), {0, 1})
        self.assertGreaterEqual(metrics["accuracy"], 0.90)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["log_loss"])))
        self.assertEqual(
            confusion_matrix(y_test, labels, num_classes=2).sum().item(),
            y_test.shape[0],
        )

    def test_knn_gives_expected_predictions_on_hand_checkable_points(self) -> None:
        x_train = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [3.0, 3.0], [3.0, 4.0]]
        )
        y_train = torch.tensor([[0], [0], [1], [1]])
        x_test = torch.tensor([[0.1, 0.2], [3.2, 3.2]])

        predictions = predict_knn(x_train, y_train, x_test, k=1)

        self.assertTrue(torch.equal(predictions, torch.tensor([[0], [1]])))

    def test_gaussian_naive_bayes_probabilities_and_accuracy(self) -> None:
        x_train, x_test, y_train, y_test = generate_binary_classification_data(
            n_samples=240,
            seed=29,
        )
        x_train, x_test, _, _ = standardize_from_train(x_train, x_test)
        model = fit_gaussian_naive_bayes(x_train, y_train)
        probabilities = predict_gaussian_nb_probabilities(x_test, model)
        labels = predict_gaussian_nb_labels(x_test, model)

        self.assertEqual(probabilities.shape, (x_test.shape[0], 2))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(x_test.shape[0]),
                atol=1e-6,
            )
        )
        self.assertGreaterEqual(
            classification_metrics(y_test, labels)["accuracy"], 0.90
        )

    def test_invalid_classifier_inputs_are_rejected(self) -> None:
        x_train = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        binary_targets = torch.tensor([[0.0], [1.0]])

        with self.assertRaises(ValueError):
            predict_knn(x_train, binary_targets, x_train, k=0)
        with self.assertRaises(ValueError):
            train_logistic_regression(
                x_train,
                torch.tensor([[0.0], [2.0]]),
                epochs=1,
            )
        with self.assertRaises(ValueError):
            fit_gaussian_naive_bayes(x_train, torch.zeros((2, 1)))
        with self.assertRaises(ValueError):
            confusion_matrix(binary_targets, torch.tensor([[0], [2]]), 2)


if __name__ == "__main__":
    unittest.main()
