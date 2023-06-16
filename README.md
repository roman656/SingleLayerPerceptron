# SingleLayerPerceptron
Обучение однослойного перцептрона дельта-правилом.

## Требования

- Использовать пороговую функцию активации
- Первоначальные веса – вещественные числа от -1.0 до 1.0
- Величина коэффициента обучения: 0,05; 0.1; 0.25; 0.5; 0.75; 0.9
- Провести 6 экспериментов (с разными коэффициентами обучения)
- Количество эпох обучения: 20
- Вывод результатов обучения по эпохам – среднеквадратичная ошибка
- Построить графики по каждому эксперименту с изменением значения ошибки в процессе обучения по эпохам (по оси X – эпохи, по оси Y – значения среднеквадратичной ошибки)

## Обучающая выборка

В качестве обучающей выборки используется таблица истинности для дизъюнкции.
Входной слой: 2 нейрона и нейрон смещения (bias).
Выходной слой: 1 нейрон.

| A | B | A V B |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

## Пример результатов обучения

![Пример графика, отражающего результаты обучения.](https://github.com/roman656/SingleLayerPerceptron/blob/main/result.png)
