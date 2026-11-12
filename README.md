# Метод k-средних

Реализация кластеризации на Python: класс `KMeans`, генерация случайных данных, CLI, график кластеров и метод локтя. Алгоритм написан без NumPy.

## Установка

```bash
python -m pip install -r requirements.txt
```

## Запуск

```bash
python kmeans.py --samples 300 --k 3 --iter 100 --seed 42
python kmeans.py --samples 300 --k 3 --iter 100 --seed 42 --elbow
```

`--elbow` строит график inertia для разных значений `k`.

## Тесты

```bash
python -m unittest discover -v
```
