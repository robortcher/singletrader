import random
import numpy as np
import pandas as pd

# 定义股票收益率数据
stock_data = pd.read_csv(r'D:\projects\singletrader_pro\samples\factor_mining\tamplate02/stock_data.csv',index_col=0)
returns = stock_data.pct_change().dropna()

# 定义遗传算法的参数
population_size = 100 # 种群大小
num_factors = 10 # 因子个数
elite_size = 10 # 精英个体数量
mutation_rate = 0.1 # 变异率
generations = 50 # 迭代次数

# 定义适应度函数
def fitness(individual, returns):
    factor_returns = returns.iloc[:, individual].mean(axis=1)
    fitness = factor_returns.sum()
    return fitness

# 定义交叉操作
def crossover(parent_1, parent_2):
    child = []
    for i in range(num_factors):
        if random.random() > 0.5:
            child.append(parent_1[i])
        else:
            child.append(parent_2[i])
    return child

# 定义变异操作
def mutate(individual):
    for i in range(num_factors):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, num_factors-1)
    return individual

# 初始化种群
population = []
for i in range(population_size):
    individual = [random.randint(0, num_factors-1) for _ in range(num_factors)]
    population.append(individual)

# 迭代
for i in range(generations):
    # 计算适应度
    fitness_scores = []
    for individual in population:
        fitness_score = fitness(individual, returns)
        fitness_scores.append(fitness_score)
    # 选择精英个体
    elite_indices = np.argsort(fitness_scores)[::-1][:elite_size]
    elite_population = [population[i] for i in elite_indices]
    # 生成新个体
    new_population = []
    while len(new_population) < population_size:
        parent_1 = random.choice(elite_population)
        parent_2 = random.choice(elite_population)
        child = crossover(parent_1, parent_2)
        child = mutate(child)
        new_population.append(child)
    population = new_population

# 输出最优个体
best_individual = elite_population[0]
best_factor_indices = [i for i, val in enumerate(best_individual) if val == 1]
print("Best factors:", best_factor_indices)
