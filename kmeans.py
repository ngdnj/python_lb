from __future__ import annotations

import math
import random
import argparse
from dataclasses import dataclass
from numbers import Real
from typing import Iterable


Point = tuple[float, ...]


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} должен быть положительным целым числом")


def _nonnegative_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} должен быть конечным неотрицательным числом")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} слишком велик") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} должен быть конечным неотрицательным числом")
    return value


def _points(data: Iterable[Iterable[float]]) -> list[Point]:
    """Проверяет данные и копирует их, не меняя входную коллекцию."""
    result = []
    try:
        for row in data:
            point = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("Координаты должны быть конечными числами")
                coordinate = float(value)
                if not math.isfinite(coordinate):
                    raise ValueError("Координаты должны быть конечными числами")
                point.append(coordinate)
            if not point or (result and len(point) != len(result[0])):
                raise ValueError("Точки должны иметь одинаковую ненулевую размерность")
            result.append(tuple(point))
    except TypeError as exc:
        raise ValueError("Ожидается коллекция точек с числовыми координатами") from exc
    except OverflowError as exc:
        raise ValueError("Слишком большие координаты: нормализуйте данные") from exc
    if not result:
        raise ValueError("Данные не должны быть пустыми")
    return result


def _distance_squared(a: Point, b: Point) -> float:
    value = sum((x - y) * (x - y) for x, y in zip(a, b))
    if not math.isfinite(value):
        raise ValueError("Слишком большие координаты: нормализуйте данные")
    return value


@dataclass(frozen=True)
class KMeansResult:
    """Метки соответствуют порядку точек; inertia — сумма квадратов расстояний."""

    centers: tuple[Point, ...]
    labels: tuple[int, ...]
    inertia: float
    n_iter: int
    converged: bool


def kmeans(
    data: Iterable[Iterable[float]],
    k: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
    seed: int | None = None,
) -> KMeansResult:
    """Кластеризует точки алгоритмом Ллойда с евклидовым расстоянием.

    Начальные центры выбираются из разных строк данных случайно. Из n_init
    запусков выбирается результат с минимальной inertia. Пустой кластер
    сохраняет предыдущий центр. При равном расстоянии выбирается первый центр.
    tol задаёт максимально допустимое перемещение каждого центра.
    """
    points = _points(data)
    _positive_int(k, "k")
    _positive_int(max_iter, "max_iter")
    _positive_int(n_init, "n_init")
    tolerance = _nonnegative_number(tol, "tol")
    if k > len(points):
        raise ValueError("k не может превышать число точек")
    rng = random.Random(seed)
    best = None
    dimension = len(points[0])

    def assign(centers: list[Point]) -> tuple[int, ...]:
        return tuple(
            min(range(k), key=lambda j: _distance_squared(point, centers[j]))
            for point in points
        )

    for _ in range(n_init):
        centers = rng.sample(points, k)
        converged = False
        for iteration in range(1, max_iter + 1):
            labels = assign(centers)
            groups: list[list[Point]] = [[] for _ in range(k)]
            for point, label in zip(points, labels):
                groups[label].append(point)
            # Деление перед суммированием уменьшает риск переполнения суммы.
            updated = []
            for j, group in enumerate(groups):
                if group:
                    center = tuple(
                        math.fsum(p[d] / len(group) for p in group)
                        for d in range(dimension)
                    )
                else:
                    center = centers[j]
                updated.append(center)
            shift = max(
                math.sqrt(_distance_squared(a, b))
                for a, b in zip(centers, updated)
            )
            centers = updated
            if shift <= tolerance:
                converged = True
                break
        # После последнего сдвига пересчитываем метки и целевую функцию.
        labels = assign(centers)
        try:
            inertia = math.fsum(
                _distance_squared(p, centers[label])
                for p, label in zip(points, labels)
            )
        except OverflowError as exc:
            raise ValueError(
                "Слишком большие координаты: нормализуйте данные"
            ) from exc
        result = KMeansResult(tuple(centers), labels, inertia, iteration, converged)
        if best is None or result.inertia < best.inertia:
            best = result
    assert best is not None
    return best


def make_blobs(
    centers: Iterable[Iterable[float]],
    samples_per_cluster: int = 50,
    *,
    std: float = 0.5,
    seed: int | None = None,
) -> tuple[list[Point], list[int]]:
    """Генерирует перемешанные гауссовы облака и их исходные метки."""
    checked = _points(centers)
    _positive_int(samples_per_cluster, "samples_per_cluster")
    spread = _nonnegative_number(std, "std")
    rng = random.Random(seed)
    samples = [
        (tuple(rng.gauss(coordinate, spread) for coordinate in center), label)
        for label, center in enumerate(checked)
        for _ in range(samples_per_cluster)
    ]
    rng.shuffle(samples)
    points = _points(point for point, _ in samples)
    return points, [label for _, label in samples]


class KMeans:
    """Обучаемая модель k-средних с интерфейсом fit/predict."""

    def __init__(self, n_clusters: int = 3, max_iter: int = 100,
                 seed: int | None = None):
        _positive_int(n_clusters, "n_clusters")
        _positive_int(max_iter, "max_iter")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.cluster_centers_: tuple[Point, ...] | None = None
        self.labels_: tuple[int, ...] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _init_centroids(self, X: list[Point]) -> list[Point]:
        if self.n_clusters > len(X):
            raise ValueError("n_clusters не может превышать число точек")
        return random.Random(self.seed).sample(X, self.n_clusters)

    def _assign(self, X: list[Point]) -> tuple[int, ...]:
        if self.cluster_centers_ is None:
            raise RuntimeError("Сначала вызовите fit")
        return tuple(min(range(self.n_clusters),
                         key=lambda j: _distance_squared(p, self.cluster_centers_[j]))
                     for p in X)

    def _update(self, X: list[Point], labels: tuple[int, ...]) -> tuple[Point, ...]:
        assert self.cluster_centers_ is not None
        groups = [[p for p, label in zip(X, labels) if label == j]
                  for j in range(self.n_clusters)]
        dimension = len(X[0])
        return tuple(
            tuple(math.fsum(p[d] / len(group) for p in group)
                  for d in range(dimension)) if group else self.cluster_centers_[j]
            for j, group in enumerate(groups)
        )

    def fit(self, X: Iterable[Iterable[float]]) -> tuple[int, ...]:
        points = _points(X)
        self.cluster_centers_ = tuple(self._init_centroids(points))
        for iteration in range(1, self.max_iter + 1):
            labels = self._assign(points)
            updated = self._update(points, labels)
            shift = max(math.sqrt(_distance_squared(a, b))
                        for a, b in zip(self.cluster_centers_, updated))
            self.cluster_centers_ = updated
            if shift <= 1e-4:
                break
        self.labels_ = self._assign(points)
        self.inertia_ = self.inertia(points, self.labels_)
        self.n_iter_ = iteration
        return self.labels_

    def predict(self, X: Iterable[Iterable[float]]) -> tuple[int, ...]:
        points = _points(X)
        if self.cluster_centers_ is None:
            raise RuntimeError("Сначала вызовите fit")
        if len(points[0]) != len(self.cluster_centers_[0]):
            raise ValueError("Размерность точек не совпадает с обучающими данными")
        return self._assign(points)

    def inertia(self, X: Iterable[Iterable[float]],
                labels: Iterable[int]) -> float:
        points = _points(X)
        labels = tuple(labels)
        if self.cluster_centers_ is None:
            raise RuntimeError("Сначала вызовите fit")
        if len(points) != len(labels) or any(
            isinstance(label, bool) or not isinstance(label, int)
            or not 0 <= label < self.n_clusters for label in labels
        ):
            raise ValueError("Нужна метка кластера для каждой точки")
        try:
            return math.fsum(_distance_squared(p, self.cluster_centers_[label])
                             for p, label in zip(points, labels))
        except OverflowError as exc:
            raise ValueError("Слишком большие координаты: нормализуйте данные") from exc


def plot_clusters(X, labels, centers, *, title="Кластеры k-средних"):
    """Показывает двумерные точки и центры; требует matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Для графиков установите matplotlib: python -m pip install matplotlib") from exc
    points = _points(X)
    if len(points[0]) != 2 or len(points) != len(labels):
        raise ValueError("Для визуализации нужны двумерные точки и метка на каждую точку")
    for label in sorted(set(labels)):
        group = [p for p, value in zip(points, labels) if value == label]
        plt.scatter([p[0] for p in group], [p[1] for p in group], label=f"Кластер {label}")
    plt.scatter([c[0] for c in centers], [c[1] for c in centers],
                marker="X", s=200, c="black", label="Центры")
    plt.title(title)
    plt.legend()
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Кластеризация методом k-средних")
    parser.add_argument("--samples", type=int, default=300, help="общее число точек")
    parser.add_argument("--k", type=int, default=3, help="число кластеров")
    parser.add_argument("--iter", type=int, default=100, dest="max_iter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elbow", action="store_true", help="построить график локтя для k=1..10")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.k < 1 or args.max_iter < 1:
        parser.error("--samples, --k и --iter должны быть положительными")
    if args.samples < args.k:
        parser.error("--samples должно быть не меньше --k: каждому кластеру нужна точка")
    centers = [(-5, -5), (0, 5), (5, -5), (-10, 5), (10, 5), (0, -10),
               (-10, -10), (10, 10), (15, 0), (-15, 0)]
    if args.k > len(centers):
        centers.extend((20 * i, 20 * j) for i in range(-args.k, args.k + 1)
                       for j in range(-args.k, args.k + 1))
    data, _ = make_blobs(centers[:args.k], args.samples // args.k,
                         seed=args.seed)
    remainder = args.samples % args.k
    if remainder:
        extra, _ = make_blobs(centers[:remainder], 1, seed=args.seed + 1)
        data.extend(extra)
        random.Random(args.seed).shuffle(data)
    if args.elbow:
        scores = []
        for k in range(1, min(10, len(data)) + 1):
            model = KMeans(k, args.max_iter, args.seed)
            model.fit(data)
            scores.append(model.inertia_)
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            parser.error("для --elbow установите matplotlib: python -m pip install matplotlib")
        plt.plot(range(1, len(scores) + 1), scores, marker="o")
        plt.xlabel("Число кластеров k")
        plt.ylabel("Inertia")
        plt.title("Метод локтя")
        plt.xticks(range(1, len(scores) + 1))
        plt.show()
        return
    model = KMeans(args.k, args.max_iter, args.seed)
    labels = model.fit(data)
    print(f"Точек: {len(data)}, кластеров: {args.k}")
    for label in range(args.k):
        print(f"Кластер {label}: {labels.count(label)} точек")
    print(f"Inertia: {model.inertia_:.4f}")
    print(f"Итераций: {model.n_iter_}")
    plot_clusters(data, labels, model.cluster_centers_)


if __name__ == "__main__":
    main()
