"""Учебная реализация метода k-средних без сторонних зависимостей."""

from __future__ import annotations

import math
import random
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


if __name__ == "__main__":
    data, _ = make_blobs([(-5, -5), (0, 5), (5, -5)], seed=42)
    result = kmeans(data, k=3, seed=42)
    print(f"Точек: {len(data)}, кластеров: {len(result.centers)}")
    for label, center in enumerate(result.centers):
        print(f"Кластер {label}: центр={tuple(round(x, 3) for x in center)}, "
              f"точек={result.labels.count(label)}")
    print(f"Сумма квадратов расстояний: {result.inertia:.4f}")
    print(f"Итераций: {result.n_iter}, сходимость: {result.converged}")
