import random as random
import time
import copy
import sys
from math import ceil
from optimal_model.classes import UE, E2_Node

def genetic_algorithm(population, fn_fitness, E2Ns_len, gene_pool, fn_thres=None, ngen=500000, pmut=0.5):
    best_individual = min(population, key=individual_cost)
    best_individuals = []
    generations = []

    for generation in range(ngen):
        new_population = []

        for _ in range(len(population)):
            parent1 = select(1, population)[0]
            child = recombine(copy.deepcopy(parent1), copy.deepcopy(best_individual)) if random.uniform(0, 1) <= 0.5 else recombine(copy.deepcopy(best_individual), copy.deepcopy(parent1))
            child = mutate(copy.deepcopy(child), pmut, E2Ns_len)
            new_population.append([child, fn_fitness(child, E2Ns_len)])

        population = new_population

        if fn_thres is not None and real_cost(best_individual) == fn_thres:
            return best_individual

        best_individual = min(population, key=individual_cost)
        best_individuals.append(best_individual[1][0])
        generations.append(generation)

    return best_individual

def individual_cost(individual):
    return individual[1][0]

def real_cost(individual):
    return individual[1][2]


def select2(num_parents, population):
    selected_parents = []
    for _ in range(num_parents):
        selected_parents.append(min(population, key=individual_cost))
    return selected_parents

def select(num_parents, population):
    selected_parents = []
    fitness_values = [1 / individual_cost(individual) for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]

    for _ in range(num_parents):
        r = random.uniform(0, 1)
        cumulative_probability = 0.0
        for individual, probability in zip(population, probabilities):
            cumulative_probability += probability
            if r < cumulative_probability:
                selected_parents.append(individual)
                break

    return selected_parents

def recombine(parent1, parent2):
    n = len(parent1[0])
    midpoint = n // 2
    child_genes = parent1[0][:midpoint] + parent2[0][midpoint:]

    return child_genes  # Return child

def mutate(individual, mutation_prob, E2Ns_len):
    if random.uniform(0, 1) > mutation_prob:
        return individual

    n = len(individual)

    gene_to_mutate = random.randrange(0, n)
    new_chromosome =random.randrange(0, E2Ns_len-1)

    individual[gene_to_mutate][1] = new_chromosome

    return individual

def init_population(gene_pool, state_length, fn_fitness, E2Ns_len):
    population = []

    for _ in range(state_length):
        new_individual = [[gene[0], random.randint(0, E2Ns_len-1)] for gene in gene_pool.values()]
        fitness = fn_fitness(new_individual, E2Ns_len)
        population.append([new_individual, fitness])

    return population

class EvaluateEnergySaver:
    def __call__(self, solution, E2Ns_len):
        E2N_on = {str(i): 0 for i in range(E2Ns_len)}
        cost = real_cost = rf_consumption = 0

        real_cost_per_e2n = 12.9 + (1 / 0.388)  # RF consumption + E2_POWER / Power_amp_efficiency
        rf_consumption_per_e2n = 12.9

        for connection in solution:
            E2N_on[str(connection[1])] += 1

        min_active = min(value for value in E2N_on.values() if value > 0)
        weight = len(solution)

        for E2N, count in E2N_on.items():
            if count > 85:
                return sys.maxsize, E2N_on
            if count >= 1:
                cost += weight
                real_cost += real_cost_per_e2n
                rf_consumption += rf_consumption_per_e2n

        return cost + min_active, E2N_on, real_cost, rf_consumption


def define_model(UEs, E2Ns, total_BW):
    possible_values_teste = {str(i): [i, 1] for i in range(len(UEs))}

    # Initialize fitness function
    fn_fitness = EvaluateEnergySaver()

    # Population length
    population_length = 10


    real_cost_per_e2n = 12.9 + (1 / 0.388)
    fn_thres = ceil(len(possible_values_teste)/85) * real_cost_per_e2n


    E2Ns_len = len(E2Ns)

    start_time = time.time()

    # Initialize population
    population = init_population(possible_values_teste, population_length, fn_fitness, E2Ns_len)


    # Run the genetic algorithm
    data = genetic_algorithm(
        population,
        fn_fitness,
        E2Ns_len,
        gene_pool=possible_values_teste,
        ngen=2000,
        pmut=0.5
    )

    RF_energy = data[1][3]
    total_energy = data[1][2]

    connections = {key: value for key, value in data[0]}

    E2N_info = {
        int(key): {'bandwidth': value, 'power': 1}
        for key, value in data[1][1].items() if value != 0
    }

    used_BW = 100/85 * len(possible_values_teste)

    users_TP = []
    for i in range(len(possible_values_teste)):
        users_TP.append(9.667941188510181e-13)

    solution = {
            "max_BW": total_BW,
            "used_BW": used_BW,
            "users_TP": users_TP,
            "users_PW": [ue.channel_gain for ue in UEs],
            "total_energy": data[1][2],
            "RF_energy": float(RF_energy),
            "PW_energy": float(total_energy) - float(RF_energy)


    }

    runtime = time.time() - start_time
    print(f"\nRuntime: {runtime}")
    print(f"\nBest total Energy: {fn_thres}")
    print(f"\nTotal Energy: {total_energy}")
    print(f"\nRF Energy: {RF_energy}")
    print(f"\nPW Energy: {float(total_energy) - float(RF_energy)}")
    return [connections, E2N_info, solution]

def run_model(input_E2N, input_UE, total_BW):    
    UEs = []
    for user in input_UE["users"]:
        UEs.append(UE(user["ID"], 
                      user["demand"], 
                      user["channel_gain"]))

    E2Ns = []    
    for E2N in input_E2N["E2_nodes"]:
        E2Ns.append(E2_Node(E2N["ID"], 
                            E2N["bandwidth"],
                            E2N["max_power"],
                            E2N["RF_consumption"],
                            E2N["Power_amp_efficiency"]))
    
    total_BW = 0
    for e2 in E2Ns:
        total_BW += e2.BW
    return define_model(UEs=UEs, E2Ns=E2Ns,total_BW=total_BW)