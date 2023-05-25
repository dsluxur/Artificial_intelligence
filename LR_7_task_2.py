import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, benchmarks, \
        cma, creator, tools

# Функція створення панелі інструментів
def create_toolbox(strategy):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list,
fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", benchmarks.rastrigin)

    # Задання генератору випадкових чисел
    np.random.seed(7)

    toolbox.register("generate", strategy.generate,
                     creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox


if __name__ == "__main__":
    # Розмір проблеми
    num_individuals = 10
    num_generations = 125

    # Створення стратегії за допомогою алгоритму CMA-ES
    strategy = cma.Strategy(centroid=[5.0] * num_individuals,
                            sigma=5.0,
                            lambda_=20 * num_individuals)

    # Створення панелі інструментів на основі наведеної вище стратегії
    toolbox = create_toolbox(strategy)

    # Створення обєкту HallOfFame
    hall_of_fame = tools.HallOfFame(1)

    # Реєстрація відповідної статистики
    stats = tools.Statistics(lambda x: x.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Об'єкти, які збиратимуть дані
    sigma = np.ndarray((num_generations, 1))
    axis_ratio = np.ndarray((num_generations, 1))
    diagD = np.ndarray((num_generations, num_individuals))
    fbest = np.ndarray((num_generations, 1))
    best = np.ndarray((num_generations, num_individuals))
    std = np.ndarray((num_generations, num_individuals))

    for gen in range(num_generations):
        # Створюємо нову популяцію
        population = toolbox.generate()

        # Оцінка індивідуумів
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Оновлення стратегії за допомогою оцінених індивідуумів
        toolbox.update(population)

        # Оновлення HallOfFame та статистики для розраховуємої в даний момент популяції
        hall_of_fame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)

        print(logbook.stream)

        # Збереження данних для побудови графіків
        sigma[gen] = strategy.sigma
        axis_ratio[gen] = max(strategy.diagD) ** 2 / min(strategy.diagD) ** 2
        diagD[gen, :num_individuals] = strategy.diagD ** 2
        fbest[gen] = hall_of_fame[0].fitness.values
        best[gen, :num_individuals] = hall_of_fame[0]
        std[gen, :num_individuals] = np.std(population, axis=0)

    # На осі абсцисс буде відкладено кількість оцінок
    x = list(range(0, strategy.lambda_ * num_generations,
                   strategy.lambda_))
    avg, max_, min_ = logbook.select("avg", "max", "min")
    plt.figure()
    plt.semilogy(x, avg, "--b")
    plt.semilogy(x, max_, "--b")
    plt.semilogy(x, min_, "-b")
    plt.semilogy(x, fbest, "-c")
    plt.semilogy(x, sigma, "-g")
    plt.semilogy(x, axis_ratio, "-r")
    plt.grid(True)
    plt.title("Синій: f-значення, зелений: sigma, червоний: axisratio")

    #Побудова графіку ходу процесу
    plt.figure()
    plt.plot(x, best)
    plt.grid(True)
    plt.title("Обєктні змінні")
    plt.figure()
    plt.semilogy(x, diagD)
    plt.grid(True)
    plt.title("Масштабування (Всі основні осі)")
    plt.figure()
    plt.semilogy(x, std)
    plt.grid(True)
    plt.title("Стандартне відхилення по всіх координатах")

    plt.show()





