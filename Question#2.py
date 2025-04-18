import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters from the problem
task_times = {
    'Task 1': 5,
    'Task 2': 8,
    'Task 3': 4,
    'Task 4': 7,
    'Task 5': 6,
    'Task 6': 3,
    'Task 7': 9
}

facility_capacities = {
    'Facility 1': 24,
    'Facility 2': 30,
    'Facility 3': 25
}

cost_matrix = {
    'Task 1': {'Facility 1': 10, 'Facility 2': 12, 'Facility 3': 15},
    'Task 2': {'Facility 1': 15, 'Facility 2': 14, 'Facility 3': 16},
    'Task 3': {'Facility 1': 8, 'Facility 2': 9, 'Facility 3': 7},
    'Task 4': {'Facility 1': 12, 'Facility 2': 10, 'Facility 3': 13},
    'Task 5': {'Facility 1': 14, 'Facility 2': 11, 'Facility 3': 12},
    'Task 6': {'Facility 1': 9, 'Facility 2': 8, 'Facility 3': 10},
    'Task 7': {'Facility 1': 11, 'Facility 2': 12, 'Facility 3': 9}
}

# GA parameters
POPULATION_SIZE = 6
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
MAX_GENERATIONS = 100

facilities = list(facility_capacities.keys())
tasks = list(task_times.keys())

# Helper functions
def create_individual():
    """Create a random assignment of tasks to facilities."""
    return {task: random.choice(facilities) for task in tasks}

def initial_population(size):
    """Create initial population of random individuals."""
    return [create_individual() for _ in range(size)]

def calculate_facility_load(individual):
    """Calculate the load (hours) for each facility."""
    loads = {facility: 0 for facility in facilities}
    for task, facility in individual.items():
        loads[facility] += task_times[task]
    return loads

def calculate_total_cost(individual):
    """Calculate the total cost for the assignment."""
    total_cost = 0
    for task, facility in individual.items():
        cost_per_hour = cost_matrix[task][facility]
        hours = task_times[task]
        total_cost += cost_per_hour * hours
    return total_cost

def calculate_fitness(individual):
    """Calculate fitness value with penalty for capacity violations."""
    loads = calculate_facility_load(individual)
    penalty = 0
    
    # Apply penalties for exceeding capacity
    for facility, load in loads.items():
        if load > facility_capacities[facility]:
            penalty += (load - facility_capacities[facility]) * 100
    
    # Total cost as the base fitness
    cost = calculate_total_cost(individual)
    
    # Return inverse of (cost + penalty) since we want to maximize fitness
    return 1.0 / (cost + penalty + 1)  # Add 1 to avoid division by zero

def roulette_wheel_selection(population, fitness_values):
    """Select an individual using roulette wheel selection."""
    total_fitness = sum(fitness_values)
    selection_point = random.uniform(0, total_fitness)
    current = 0
    
    for i, fitness in enumerate(fitness_values):
        current += fitness
        if current >= selection_point:
            return population[i]
    
    return population[-1]  # Fallback if rounding issues

def one_point_crossover(parent1, parent2):
    """Perform one-point crossover between two parents."""
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    
    # Choose a random crossover point
    crossover_point = random.randint(1, len(tasks) - 1)
    tasks_list = list(tasks)
    
    # Create offspring
    offspring1 = {}
    offspring2 = {}
    
    for i, task in enumerate(tasks_list):
        if i < crossover_point:
            offspring1[task] = parent1[task]
            offspring2[task] = parent2[task]
        else:
            offspring1[task] = parent2[task]
            offspring2[task] = parent1[task]
    
    return offspring1, offspring2

def swap_mutation(individual):
    """Perform swap mutation on an individual."""
    if random.random() > MUTATION_RATE:
        return individual
    
    mutated = individual.copy()
    task = random.choice(tasks)
    # Choose a different facility
    current_facility = mutated[task]
    other_facilities = [f for f in facilities if f != current_facility]
    new_facility = random.choice(other_facilities)
    mutated[task] = new_facility
    
    return mutated

def genetic_algorithm():
    """Main genetic algorithm function."""
    # Initialize population
    population = initial_population(POPULATION_SIZE)
    
    best_fitness_history = []
    avg_fitness_history = []
    best_individual = None
    best_fitness = -1
    
    for generation in range(MAX_GENERATIONS):
        # Calculate fitness for each individual
        fitness_values = [calculate_fitness(individual) for individual in population]
        
        # Track the best individual
        max_fitness_idx = fitness_values.index(max(fitness_values))
        current_best = population[max_fitness_idx]
        current_best_fitness = fitness_values[max_fitness_idx]
        
        if current_best_fitness > best_fitness:
            best_individual = current_best.copy()
            best_fitness = current_best_fitness
        
        # Record history
        best_fitness_history.append(1.0 / best_fitness - 1)  # Convert back to cost
        avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
        
        # Create new population
        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(population[max_fitness_idx].copy())
        
        # Create rest of the new population
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            
            # Crossover
            offspring1, offspring2 = one_point_crossover(parent1, parent2)
            
            # Mutation
            offspring1 = swap_mutation(offspring1)
            offspring2 = swap_mutation(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(offspring2)
        
        population = new_population
    
    return best_individual, best_fitness_history, avg_fitness_history

# Run the algorithm
best_solution, best_fitness_history, avg_fitness_history = genetic_algorithm()

# Calculate and display the results
loads = calculate_facility_load(best_solution)
total_cost = calculate_total_cost(best_solution)

print("Best Assignment:")
for task, facility in best_solution.items():
    print(f"{task} -> {facility}")

print("\nFacility Loads (hours):")
for facility, load in loads.items():
    print(f"{facility}: {load}/{facility_capacities[facility]}")

print(f"\nTotal Cost: {total_cost}")

# Check if the solution is feasible
is_feasible = True
for facility, load in loads.items():
    if load > facility_capacities[facility]:
        is_feasible = False
        break

print(f"\nSolution is {'feasible' if is_feasible else 'not feasible'}")

# Plot the fitness history
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history, label='Best Solution Cost')
plt.xlabel('Generation')
plt.ylabel('Cost (lower is better)')
plt.title('Best Solution Cost Over Generations')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the assignments
task_to_idx = {task: i for i, task in enumerate(tasks)}
facility_to_idx = {facility: i for i, facility in enumerate(facilities)}

assignment_matrix = np.zeros((len(tasks), len(facilities)))
for task, facility in best_solution.items():
    assignment_matrix[task_to_idx[task], facility_to_idx[facility]] = 1

plt.figure(figsize=(10, 6))
plt.imshow(assignment_matrix, cmap='Blues')
plt.xticks(range(len(facilities)), facilities)
plt.yticks(range(len(tasks)), tasks)
plt.colorbar()
plt.title('Task Assignment to Facilities')
plt.ylabel('Tasks')
plt.xlabel('Facilities')

# Add task times as text
for i, task in enumerate(tasks):
    for j, facility in enumerate(facilities):
        if assignment_matrix[i, j] == 1:
            plt.text(j, i, f"{task_times[task]}h", ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# Additional visualization of costs and facility utilization
facility_costs = {facility: 0 for facility in facilities}
for task, facility in best_solution.items():
    facility_costs[facility] += cost_matrix[task][facility] * task_times[task]

plt.figure(figsize=(12, 5))

# Plot 1: Facility Utilization
plt.subplot(1, 2, 1)
utilization = []
for facility in facilities:
    utilization.append(loads[facility] / facility_capacities[facility] * 100)

plt.bar(facilities, utilization)
plt.axhline(y=100, color='r', linestyle='--')
plt.ylabel('Utilization (%)')
plt.title('Facility Utilization')
for i, v in enumerate(utilization):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center')

# Plot 2: Facility Costs
plt.subplot(1, 2, 2)
plt.bar(facilities, [facility_costs[f] for f in facilities])
plt.ylabel('Cost')
plt.title('Cost per Facility')
for i, v in enumerate([facility_costs[f] for f in facilities]):
    plt.text(i, v + 5, f"{v}", ha='center')

plt.tight_layout()
plt.show()