import numpy as np
import matplotlib.pyplot as plt


def activation_function(weighted_sum):
    return np.where(weighted_sum >= 0, 1, 0)


def get_weighted_sum(inputs, weights):
    return inputs @ weights


def get_weights_delta(inputs, rate, real_output, expected_output):
    return rate * (expected_output - real_output) * inputs


def get_total_energy_error(real_output, expected_output):
    return 0.5 * (expected_output - real_output)**2


def draw_results(rms_error_energy_values, epochs_amount, experiments_amount):
    x = np.linspace(start=1, stop=epochs_amount, num=epochs_amount)

    figure = plt.figure()

    for index in range(experiments_amount):
        plt.plot(x, rms_error_energy_values[index], label=f'Эксперимент №{index + 1}')

    plt.xlabel("Эпохи")
    plt.ylabel("Значения среднеквадратичной ошибки")
    plt.grid(True)
    plt.xticks(range(1, epochs_amount + 1))
    plt.legend(loc='upper right')

    figure.tight_layout()
    figure.savefig(fname='result.png')


if __name__ == '__main__':
    epochs_amount = 20
    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    experiments_amount = len(learning_rates)
    rms_error_energy_values = np.zeros([experiments_amount, epochs_amount])
    expected_outputs = np.array([0, 1, 1, 1])
    training_inputs = np.array([[1, 0, 0],
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 1, 1]])  # 1й столбец - bias.
    iterations_amount = len(expected_outputs)
    inputs_amount = training_inputs.shape[1]
    start_weights = np.random.uniform(-1, 1, inputs_amount)
    experiment_index = 0

    for learning_rate in learning_rates:
        print(f'\nЭксперимент №{experiment_index + 1} с коэффициентом обучения: {learning_rate}')
        current_weights = start_weights.copy()

        for epoch in range(epochs_amount):
            total_energy_error = 0

            for iteration in range(iterations_amount):
                output = activation_function(get_weighted_sum(training_inputs[iteration], current_weights))
                current_weights += get_weights_delta(training_inputs[iteration], learning_rate, output,
                                                     expected_outputs[iteration])
                total_energy_error += get_total_energy_error(output, expected_outputs[iteration])

            rms_error_energy = total_energy_error / iterations_amount
            rms_error_energy_values[experiment_index][epoch] = rms_error_energy
            print(f'Энергия среднеквадратичной ошибки на {epoch + 1} эпохе: {rms_error_energy}')

        print(f'Первоначальные веса: {start_weights}')
        print(f'Итоговые веса: {current_weights}')
        result_output = activation_function(get_weighted_sum(training_inputs, current_weights))
        result_output.shape = [len(result_output), 1]
        print(f'Результат:\n{result_output}')
        experiment_index += 1

    draw_results(rms_error_energy_values, epochs_amount, experiments_amount)
