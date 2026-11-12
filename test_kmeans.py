"""Запуск: python -m unittest discover -v."""

import copy
import io
import math
import random
import runpy
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path

from kmeans import kmeans, make_blobs


# Один набор проверяет общий контракт данных у обеих публичных функций.
INVALID_DATA = (
    [], None, 42, [1, 2], [None], [()], [(1,), (1, 2)],
    [(True,)], [("1",)], [(None,)], [(1j,)],
    [(math.nan,)], [(math.inf,)], [(-math.inf,)], [(10**400,)],
)
INVALID_COUNTS = (0, -1, 1.5, True, "2", None)
INVALID_SPREADS = (-1, math.nan, math.inf, -math.inf, True, "1", None, 10**400)


def partition(labels):
    """Разбиение на группы индексов, не зависящее от номеров кластеров."""
    return {frozenset(i for i, value in enumerate(labels) if value == label)
            for label in set(labels)}


class KMeansTests(unittest.TestCase):
    def test_demo_runs_and_reports_result(self):
        output = io.StringIO()
        with redirect_stdout(output):
            runpy.run_path(str(Path(__file__).with_name("kmeans.py")),
                           run_name="__main__")
        self.assertIn("Точек: 150, кластеров: 3", output.getvalue())
        self.assertIn("Сумма квадратов расстояний:", output.getvalue())
        self.assertIn("сходимость: True", output.getvalue())

    def assert_consistent(self, data, result):
        self.assertEqual(len(result.labels), len(data))
        self.assertTrue(math.isfinite(result.inertia))
        self.assertGreaterEqual(result.inertia, 0)
        distances = []
        for point, label in zip(data, result.labels):
            self.assertIn(label, range(len(result.centers)))
            squared = [math.dist(point, c)**2 for c in result.centers]
            self.assertAlmostEqual(squared[label], min(squared))
            distances.append(squared[label])
        self.assertAlmostEqual(result.inertia, math.fsum(distances))

    def test_known_two_clusters(self):
        data = [(0, 0), (0, 2), (10, 10), (10, 12)]
        result = kmeans(data, 2, seed=42)
        self.assertEqual(partition(result.labels), {frozenset((0, 1)),
                                                    frozenset((2, 3))})
        self.assertEqual(set(result.centers), {(0, 1), (10, 11)})
        self.assertAlmostEqual(result.inertia, 4)
        self.assertTrue(result.converged)
        self.assert_consistent(data, result)

    def test_one_cluster_mean_and_inertia(self):
        result = kmeans([(0,), (2,), (4,)], 1, seed=1)
        self.assertEqual(result.centers, ((2.0,),))
        self.assertEqual(result.labels, (0, 0, 0))
        self.assertAlmostEqual(result.inertia, 8)

    def test_single_point(self):
        result = kmeans([(3, -7)], 1, seed=1)
        self.assertEqual(result.centers, ((3.0, -7.0),))
        self.assertEqual(result.labels, (0,))
        self.assertEqual(result.inertia, 0)
        self.assertEqual(result.n_iter, 1)
        self.assertTrue(result.converged)

    def test_k_equals_number_of_distinct_points(self):
        data = [(0,), (3,), (8,)]
        result = kmeans(data, len(data), seed=3)
        self.assertEqual(len(set(result.labels)), 3)
        self.assertEqual(set(result.centers), set(data))
        self.assertEqual(result.inertia, 0)

    def test_identical_points_empty_clusters_and_ties(self):
        result = kmeans([(2, 2)] * 4, 3, seed=2)
        self.assertEqual(result.centers, ((2.0, 2.0),) * 3)
        self.assertEqual(result.labels, (0,) * 4)
        self.assertEqual(result.inertia, 0)
        self.assertTrue(result.converged)

    def test_duplicate_points(self):
        data = [(0,)] * 3 + [(10,)] * 3
        result = kmeans(data, 2, seed=42)
        self.assertEqual(len(set(result.labels)), 2)
        self.assertEqual(result.inertia, 0)

    def test_arbitrary_dimension(self):
        for dimension in (1, 2, 3, 10):
            with self.subTest(dimension=dimension):
                data = [tuple([0] * dimension), tuple([2] * dimension)]
                result = kmeans(data, 1, seed=1)
                self.assertEqual(result.centers, (tuple([1.0] * dimension),))
                self.assertAlmostEqual(result.inertia, 2 * dimension)

    def test_input_is_not_modified(self):
        data = [[0, 0], [1, 2], [8, 9]]
        original = copy.deepcopy(data)
        kmeans(data, 2, seed=1)
        self.assertEqual(data, original)

    def test_generators_are_supported(self):
        data = [(0, 2), (2, 4)]
        expected = kmeans(data, 1, seed=4)
        self.assertEqual(kmeans((iter(p) for p in data), 1, seed=4), expected)

    def test_result_is_immutable(self):
        result = kmeans([(0,)], 1)
        with self.assertRaises(FrozenInstanceError):
            result.inertia = 1
        with self.assertRaises(TypeError):
            result.centers[0][0] = 1

    def test_seed_reproducibility(self):
        data, _ = make_blobs([(0, 0), (2, 2)], seed=8)
        self.assertEqual(kmeans(data, 2, seed=7), kmeans(data, 2, seed=7))

    def test_global_random_state_unchanged(self):
        before = random.getstate()
        kmeans([(0,), (1,), (5,)], 2, seed=1)
        self.assertEqual(random.getstate(), before)

    def test_iteration_limit(self):
        result = kmeans([(0,), (2,)], 1, max_iter=1, n_init=1, tol=0, seed=1)
        self.assertEqual(result.n_iter, 1)
        self.assertFalse(result.converged)
        self.assertEqual(result.centers, ((1.0,),))
        self.assert_consistent([(0,), (2,)], result)

    def test_zero_tolerance_reaches_stationary_centers(self):
        result = kmeans([(0,), (2,)], 1, n_init=1, tol=0, seed=1)
        self.assertTrue(result.converged)
        self.assertEqual(result.n_iter, 2)

    def test_large_tolerance_stops_after_first_update(self):
        result = kmeans([(0,), (2,)], 1, tol=10, seed=1)
        self.assertEqual(result.n_iter, 1)
        self.assertTrue(result.converged)

    def test_final_labels_match_centers_even_on_early_stop(self):
        data = [(0,), (1,), (2,), (4,), (10,), (11,)]
        for seed in range(10):
            for options in ({"max_iter": 1}, {"tol": 100}):
                with self.subTest(seed=seed, options=options):
                    result = kmeans(data, 2, n_init=1, seed=seed, **options)
                    self.assert_consistent(data, result)

    def test_more_restarts_cannot_worsen_objective(self):
        data, _ = make_blobs([(0, 0), (2, 2), (5, 0)], std=1, seed=5)
        for seed in range(5):
            with self.subTest(seed=seed):
                one = kmeans(data, 3, n_init=1, seed=seed)
                many = kmeans(data, 3, n_init=8, seed=seed)
                self.assertLessEqual(many.inertia, one.inertia)

    def test_more_iterations_cannot_worsen_objective(self):
        data, _ = make_blobs([(0, 0), (3, 3)], std=2, seed=15)
        previous = math.inf
        for limit in (1, 2, 3, 10, 100):
            result = kmeans(data, 2, n_init=1, max_iter=limit, tol=0, seed=9)
            self.assertLessEqual(result.inertia, previous + 1e-10)
            previous = result.inertia

    def test_random_separated_blobs(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                data, truth = make_blobs(
                    [(-20, -20), (0, 20), (20, -20)],
                    samples_per_cluster=20, std=0.2, seed=seed,
                )
                result = kmeans(data, 3, seed=seed, tol=0)
                self.assertEqual(partition(result.labels), partition(truth))
                self.assertTrue(result.converged)
                self.assert_consistent(data, result)
                for label, center in enumerate(result.centers):
                    members = [p for p, c in zip(data, result.labels) if c == label]
                    for d, coordinate in enumerate(center):
                        self.assertAlmostEqual(
                            coordinate, sum(p[d] for p in members) / len(members)
                        )

    def test_translation_and_scaling(self):
        data = [(0, 0), (1, 2), (10, 10), (11, 12)]
        original = kmeans(data, 2, seed=42, tol=0)
        transformed = [(3 * x + 7, 3 * y - 4) for x, y in data]
        result = kmeans(transformed, 2, seed=42, tol=0)
        self.assertEqual(partition(result.labels), partition(original.labels))
        self.assertAlmostEqual(result.inertia, original.inertia * 9)
        self.assertEqual(set(result.centers),
                         {(3 * x + 7, 3 * y - 4) for x, y in original.centers})

    def test_invalid_data(self):
        for data in INVALID_DATA:
            with self.subTest(data=data), self.assertRaises(ValueError):
                kmeans(data, 1)

    def test_invalid_integer_parameters(self):
        for name in ("k", "max_iter", "n_init"):
            for value in INVALID_COUNTS:
                with self.subTest(name=name, value=value):
                    options = {"k": 1, name: value}
                    with self.assertRaises(ValueError):
                        kmeans([(0,), (1,)], **options)

    def test_k_exceeds_sample_count(self):
        with self.assertRaises(ValueError):
            kmeans([(0,)], 2)

    def test_invalid_tolerance(self):
        for value in INVALID_SPREADS:
            with self.subTest(value=value), self.assertRaises(ValueError):
                kmeans([(0,)], 1, tol=value)

    def test_distance_overflow(self):
        with self.assertRaisesRegex(ValueError, "нормализуйте"):
            kmeans([(-1e308,), (1e308,)], 1, seed=1)

    def test_inertia_overflow(self):
        # Каждое расстояние конечно, но их сумма превышает диапазон float.
        data = [(0,)] + [(-8e153,)] * 2 + [(8e153,)] * 2
        with self.assertRaisesRegex(ValueError, "нормализуйте"):
            kmeans(data, 1, n_init=1, seed=2)


class MakeBlobsTests(unittest.TestCase):
    def test_shape_counts_and_finite_coordinates(self):
        data, labels = make_blobs([(0, 1, 2), (3, 4, 5)], 7, seed=1)
        self.assertEqual(len(data), 14)
        self.assertEqual(len(labels), 14)
        self.assertEqual(labels.count(0), 7)
        self.assertEqual(labels.count(1), 7)
        self.assertTrue(all(len(p) == 3 for p in data))
        self.assertTrue(all(math.isfinite(x) for p in data for x in p))

    def test_zero_spread_preserves_label_point_alignment(self):
        centers = [(1, 2), (8, 9)]
        data, labels = make_blobs(centers, 10, std=0, seed=42)
        self.assertTrue(all(p == centers[label] for p, label in zip(data, labels)))
        self.assertNotEqual(labels, sorted(labels))

    def test_reproducibility(self):
        self.assertEqual(make_blobs([(0, 0)], seed=10),
                         make_blobs([(0, 0)], seed=10))

    def test_different_seeds_generate_different_data(self):
        self.assertNotEqual(make_blobs([(0, 0)], seed=1),
                            make_blobs([(0, 0)], seed=2))

    def test_distribution_mean_and_spread(self):
        data, _ = make_blobs([(10,)], 5000, std=2, seed=42)
        mean = sum(p[0] for p in data) / len(data)
        variance = sum((p[0] - mean)**2 for p in data) / len(data)
        self.assertAlmostEqual(mean, 10, delta=0.15)
        self.assertAlmostEqual(math.sqrt(variance), 2, delta=0.15)

    def test_one_sample_per_cluster(self):
        data, labels = make_blobs([(1,), (2,)], 1, std=0, seed=1)
        self.assertEqual(set(data), {(1.0,), (2.0,)})
        self.assertEqual(set(labels), {0, 1})

    def test_generators_and_input_immutability(self):
        centers = [[0, 1], [2, 3]]
        original = copy.deepcopy(centers)
        expected = make_blobs(centers, seed=1)
        self.assertEqual(centers, original)
        self.assertEqual(make_blobs((iter(c) for c in centers), seed=1), expected)

    def test_global_random_state_unchanged(self):
        before = random.getstate()
        make_blobs([(0, 0)], seed=1)
        self.assertEqual(random.getstate(), before)

    def test_invalid_centers(self):
        for centers in INVALID_DATA:
            with self.subTest(centers=centers), self.assertRaises(ValueError):
                make_blobs(centers)

    def test_invalid_sample_count(self):
        for value in INVALID_COUNTS:
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_blobs([(0,)], value)

    def test_invalid_spread(self):
        for value in INVALID_SPREADS:
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_blobs([(0,)], std=value)

    def test_generated_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            make_blobs([(1e308,)], std=1e308, seed=1)


if __name__ == "__main__":
    unittest.main()
