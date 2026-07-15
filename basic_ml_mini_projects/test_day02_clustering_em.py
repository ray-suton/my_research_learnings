"""Deterministic checks for the Day 2 clustering and EM implementations."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from basic_ml_mini_projects.day02_clustering_em import (
    fit_dbscan,
    fit_gaussian_mixture,
    fit_kmeans,
    generate_concentric_rings,
    generate_elongated_blobs,
    generate_isotropic_blobs,
    predict_gaussian_mixture_labels,
    predict_gaussian_mixture_probabilities,
    predict_kmeans,
    run_day02_demo,
    silhouette_score,
)


CSV_HEADER = (
    "section,dataset,model,setting,n_clusters,inertia,silhouette,"
    "noise_points,initial_log_likelihood,final_log_likelihood,"
    "mean_max_responsibility,component_weights,iterations"
)


class Day02ClusteringEMTests(unittest.TestCase):
    def test_generators_are_reproducible_with_expected_shapes_and_types(
        self,
    ) -> None:
        calls = (
            (generate_isotropic_blobs, {"n_samples": 10, "seed": 7}, 10),
            (generate_elongated_blobs, {"n_samples": 11, "seed": 8}, 11),
            (
                generate_concentric_rings,
                {"points_per_ring": 7, "n_noise": 3, "seed": 9},
                17,
            ),
        )

        for generator, arguments, expected_count in calls:
            with self.subTest(generator=generator.__name__):
                first_x, first_labels = generator(**arguments)
                second_x, second_labels = generator(**arguments)

                self.assertEqual(first_x.shape, (expected_count, 2))
                self.assertEqual(first_labels.shape, (expected_count,))
                self.assertEqual(first_x.dtype, torch.float32)
                self.assertEqual(first_labels.dtype, torch.int64)
                self.assertTrue(torch.isfinite(first_x).all().item())
                self.assertTrue(torch.equal(first_x, second_x))
                self.assertTrue(torch.equal(first_labels, second_labels))

    def test_kmeans_matches_hand_calculation_and_inertia_is_non_increasing(
        self,
    ) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [10.0, 12.0]]
        )

        result = fit_kmeans(x, n_clusters=2, seed=13)
        repeated = fit_kmeans(x, n_clusters=2, seed=13)
        order = torch.argsort(result.centroids[:, 0])
        expected_centroids = torch.tensor([[0.0, 1.0], [10.0, 11.0]])

        self.assertTrue(
            torch.allclose(result.centroids[order], expected_centroids, atol=1e-5)
        )
        self.assertAlmostEqual(result.inertia_history[-1], 4.0, places=5)
        self.assertTrue(
            all(
                later <= earlier + 1e-5
                for earlier, later in zip(
                    result.inertia_history, result.inertia_history[1:]
                )
            )
        )
        self.assertTrue(torch.equal(result.labels, predict_kmeans(x, result.centroids)))
        self.assertTrue(torch.equal(result.labels, repeated.labels))
        self.assertTrue(torch.equal(result.centroids, repeated.centroids))
        self.assertEqual(result.inertia_history, repeated.inertia_history)

    def test_kmeans_recovers_empty_clusters_and_rejects_invalid_inputs(
        self,
    ) -> None:
        duplicate_x = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
        )
        result = fit_kmeans(duplicate_x, n_clusters=3, seed=3)

        self.assertTrue(torch.isfinite(result.centroids).all().item())
        self.assertTrue(torch.isfinite(torch.tensor(result.inertia_history)).all().item())
        self.assertTrue(torch.equal(result.labels, predict_kmeans(duplicate_x, result.centroids)))

        invalid_calls = (
            lambda: fit_kmeans(torch.ones(4), n_clusters=2),
            lambda: fit_kmeans(torch.tensor([[0.0, float("nan")]]), n_clusters=1),
            lambda: fit_kmeans(duplicate_x, n_clusters=0),
            lambda: fit_kmeans(duplicate_x, n_clusters=5),
            lambda: fit_kmeans(duplicate_x, n_clusters=2, max_iterations=0),
            lambda: fit_kmeans(duplicate_x, n_clusters=2, tolerance=-1e-3),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_silhouette_scores_separation_and_rejects_one_cluster(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [0.0, 0.1], [5.0, 5.0], [5.0, 5.1]]
        )
        labels = torch.tensor([0, 0, 1, 1])

        score = silhouette_score(x, labels)

        self.assertTrue(torch.isfinite(torch.tensor(score)).item())
        self.assertGreater(score, 0.8)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        with self.assertRaises(ValueError):
            silhouette_score(x, torch.zeros(4, dtype=torch.int64))

    def test_dbscan_finds_dense_groups_noise_and_validates_inputs(self) -> None:
        x = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [3.0, 3.0],
                [3.1, 3.0],
                [3.0, 3.1],
                [8.0, 8.0],
            ]
        )

        result = fit_dbscan(x, eps=0.2, min_samples=3)
        memberships = {
            frozenset(torch.where(result.labels == cluster_id)[0].tolist())
            for cluster_id in range(result.n_clusters)
        }

        self.assertEqual(memberships, {frozenset({0, 1, 2}), frozenset({3, 4, 5})})
        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.n_noise, 1)
        self.assertEqual(torch.where(result.labels == -1)[0].tolist(), [6])
        self.assertEqual(result.core_sample_mask.dtype, torch.bool)
        self.assertEqual(result.core_sample_mask.shape, (x.shape[0],))
        self.assertEqual(
            sorted(torch.unique(result.labels[result.labels >= 0]).tolist()), [0, 1]
        )

        invalid_calls = (
            lambda: fit_dbscan(x, eps=0.0, min_samples=3),
            lambda: fit_dbscan(x, eps=0.2, min_samples=0),
            lambda: fit_dbscan(torch.ones(4), eps=0.2, min_samples=2),
            lambda: fit_dbscan(
                torch.tensor([[0.0, float("inf")]]), eps=0.2, min_samples=1
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_dbscan_promotes_border_points_and_includes_eps_boundary(self) -> None:
        density_chain = torch.tensor([[0.0], [1.0], [2.0], [3.0]])

        chain_result = fit_dbscan(density_chain, eps=1.0, min_samples=3)

        self.assertEqual(chain_result.n_clusters, 1)
        self.assertEqual(chain_result.n_noise, 0)
        self.assertTrue(torch.equal(chain_result.labels, torch.zeros(4, dtype=torch.int64)))
        self.assertTrue(
            torch.equal(
                chain_result.core_sample_mask,
                torch.tensor([False, True, True, False]),
            )
        )

        exact_boundary = torch.tensor([[0.0], [1.0]])
        boundary_result = fit_dbscan(exact_boundary, eps=1.0, min_samples=2)
        self.assertEqual(boundary_result.n_clusters, 1)
        self.assertEqual(boundary_result.n_noise, 0)
        self.assertTrue(boundary_result.core_sample_mask.all().item())

    def test_gmm_invariants_means_predictions_and_likelihood(self) -> None:
        generator = torch.Generator().manual_seed(17)
        left = 0.08 * torch.randn((40, 2), generator=generator) + torch.tensor(
            [-2.0, -1.0]
        )
        right = 0.08 * torch.randn((40, 2), generator=generator) + torch.tensor(
            [2.0, 1.0]
        )
        x = torch.cat((left, right), dim=0)
        variance_floor = 1e-5

        result = fit_gaussian_mixture(
            x,
            n_components=2,
            max_iterations=50,
            tolerance=1e-5,
            variance_floor=variance_floor,
            seed=19,
        )
        probabilities = predict_gaussian_mixture_probabilities(x, result)
        labels = predict_gaussian_mixture_labels(x, result)
        ordered_means = result.means[torch.argsort(result.means[:, 0])]

        self.assertTrue((result.weights > 0.0).all().item())
        self.assertTrue(torch.allclose(result.weights.sum(), torch.tensor(1.0), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                result.responsibilities.sum(dim=1),
                torch.ones(x.shape[0]),
                atol=1e-5,
            )
        )
        self.assertTrue(torch.isfinite(result.variances).all().item())
        self.assertTrue((result.variances >= variance_floor).all().item())
        self.assertTrue(
            torch.allclose(
                ordered_means,
                torch.tensor([[-2.0, -1.0], [2.0, 1.0]]),
                atol=0.25,
            )
        )
        self.assertTrue(
            all(
                later >= earlier - 1e-3
                for earlier, later in zip(
                    result.log_likelihood_history,
                    result.log_likelihood_history[1:],
                )
            )
        )
        self.assertTrue(torch.allclose(probabilities, result.responsibilities, atol=1e-5))
        self.assertTrue(torch.equal(labels, probabilities.argmax(dim=1)))

    def test_gmm_handles_collapsed_components_and_rejects_invalid_inputs(
        self,
    ) -> None:
        repeated_x = torch.zeros((8, 2))
        variance_floor = 1e-4
        result = fit_gaussian_mixture(
            repeated_x,
            n_components=3,
            max_iterations=10,
            variance_floor=variance_floor,
            seed=23,
        )

        self.assertTrue(torch.isfinite(result.weights).all().item())
        self.assertTrue(torch.isfinite(result.means).all().item())
        self.assertTrue(torch.isfinite(result.variances).all().item())
        self.assertTrue(torch.isfinite(result.responsibilities).all().item())
        self.assertTrue((result.weights > 0.0).all().item())
        self.assertTrue(torch.allclose(result.weights.sum(), torch.tensor(1.0), atol=1e-6))
        self.assertTrue((result.variances >= variance_floor).all().item())
        self.assertTrue(
            torch.allclose(
                result.responsibilities.sum(dim=1),
                torch.ones(repeated_x.shape[0]),
                atol=1e-5,
            )
        )

        valid_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        invalid_calls = (
            lambda: fit_gaussian_mixture(valid_x, n_components=0),
            lambda: fit_gaussian_mixture(valid_x, n_components=3),
            lambda: fit_gaussian_mixture(valid_x, n_components=1, max_iterations=0),
            lambda: fit_gaussian_mixture(valid_x, n_components=1, tolerance=-1e-3),
            lambda: fit_gaussian_mixture(valid_x, n_components=1, variance_floor=0.0),
            lambda: fit_gaussian_mixture(torch.ones(4), n_components=1),
            lambda: fit_gaussian_mixture(
                torch.tensor([[0.0, float("nan")]]), n_components=1
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_demo_writes_exact_deterministic_csv_and_one_nonempty_png(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_output = Path(first_directory)
            second_output = Path(second_directory)

            first_rows = run_day02_demo(first_output)
            second_rows = run_day02_demo(second_output)
            first_csv = first_output / "day02_metrics.csv"
            second_csv = second_output / "day02_metrics.csv"

            self.assertEqual(len(first_rows), 11)
            self.assertEqual(len(second_rows), 11)
            self.assertEqual(first_csv.read_text(encoding="utf-8").splitlines()[0], CSV_HEADER)
            with first_csv.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                parsed_rows = list(reader)
                self.assertEqual(reader.fieldnames, CSV_HEADER.split(","))
            self.assertEqual(len(parsed_rows), 11)
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())

            first_files = {path.name for path in first_output.iterdir()}
            self.assertEqual(
                first_files,
                {"day02_metrics.csv", "day02_clustering_em_summary.png"},
            )
            png_files = list(first_output.glob("*.png"))
            self.assertEqual(len(png_files), 1)
            self.assertGreater(png_files[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
