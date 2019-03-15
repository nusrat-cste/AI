import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import operator

plt.ion()
plt.figure(figsize=(10,5))

def plotTSP(generation, path, points, path_distance, save, num_iters=1):

	"""
	generation: The generation number to display
	path: List of lists with the different orders in which the nodes are visited
	points: coordinates for the different nodes
	path_distance: the distance to display in the figure
	save: True if saving to final_route.png, False otherwise
	num_iters: number of paths that are in the path list

	SOURCE: https://gist.github.com/payoung/6087046

	"""
	### MOD: Brian Bennett

	plt.clf()
	plt.suptitle("Tennessee Traveling Postal Worker - Generation " + str(generation) + \
				 "\nPath Length: " + str(path_distance))
	### END MOD

	# Unpack the primary TSP path and transform it into a list of ordered
	# coordinates

	x = []; y = []
	for i in path:
		x.append(points[i][0])
		y.append(points[i][1])

	plt.plot(x, y, 'ko')

	# Set a scale for the arrow heads (there should be a reasonable default for this)
	a_scale = float(max(x))/float(2500)	# MOD: Brian Bennett
	# Draw the older paths, if provided
	if num_iters > 1:

		for i in range(1, num_iters):

			# Transform the old paths into a list of coordinates
			xi = []; yi = [];
			for j in paths[i]:
				xi.append(points[j][0])
				yi.append(points[j][1])

			plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
					head_width = a_scale, color = 'r',
					length_includes_head = True, ls = 'dashed',
					width = 0.001/float(num_iters))
			for i in range(0, len(x) - 1):
				plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
						head_width = a_scale, color = 'r', length_includes_head = True,
						ls = 'dashed', width = 0.001/float(num_iters))

	# Draw the primary path for the TSP problem
	plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
			color ='b', length_includes_head=True)
	for i in range(0,len(x)-1):
		plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
				color = 'b', length_includes_head = True)

	if save:
		plt.savefig("final_route.png")

	plt.pause(1)

class GeneticSearch:
	"""
		Class: GeneticSearch
	"""
	def __init__(self, origin, generations, points, cities, population_size, mutation_rate):
		self.population = None
		self.points = points
		self.cities = cities
		self.chromosome_size = len(self.points)
		self.generations = generations
		self.population_size = population_size
		self.mutation_rate = mutation_rate
		self.origin = origin
		self.origin_index = self.points.index(self.origin)
		self.values = []


	def print_population(self, generation, chromosomes):
		index = 0
		print("===== GENERATION %d" % generation)
		for chromosome in self.population:
			print ("Index %5d , Fitness %0.4f : %s" % (index,chromosome[1], ''.join(str(chromosome[0]))))
			index = index + 1
			if index > chromosomes:
				break


	def initialize_population(self):

		self.population = []

		# TODO: This code generates a random initial population.
		#       You may adjust this code in any way that you believe would help.
		for i in range(self.population_size):
			# replaced the code for generating random number
			individual = [x for x in range(self.chromosome_size)]
			random.shuffle(individual)
			# Move the origin_index to the front of the path
			individual.remove(self.origin_index)
			individual = [self.origin_index] + individual
			fitness = self.fitnessfcn(individual)

			# Prevent duplicate individuals in the initial population
			while [individual,fitness] in self.population:
				individual = random.sample(range(0, 10), 10)
				individual.remove(self.origin_index)
				individual = [self.origin_index] + individual
				fitness = self.fitnessfcn(individual)

			# POPULATION NODES are in the form [chromosome, fitness]
			self.population.append([individual,fitness])

		# Sort the population in descending order
		# -- "Maximize the objective function"
		self.population.sort(key=operator.itemgetter(1),reverse=True)

	def straight_line_distance(self, p1, p2):
		'''
			Return the Euclidian Distance between p1 and p2
		'''
		sld = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
		return sld


	def route_distance(self, individual):
		'''
			Determine the distance for the entire route
		'''
		distance = 0
		value = 0

		tour = individual + [self.origin_index]

		index = 0
		p1 = p2 = None
		while p2 != self.origin:
			p1 = self.points[tour[index]]
			p2 = self.points[tour[index+1]]
			distance += self.straight_line_distance(p1,p2)
			index += 1

		return distance


	def fitnessfcn(self, individual):
		'''
			Return the negative route distance so it can be maximized.
		'''
		return -self.route_distance(individual)


	def select_parents(self):
		'''
			Selects two parents from the population and returns them as a list
		'''
		# TODO: Consider a selection strategy
		selection_point = int((self.population_size*0.1))
		parents = random.sample(self.population[:selection_point], 2)
		parent1 = parents[0][0]
		parent2 = parents[1][0]
		return parent1, parent2

	def crossover(self, parent1,parent2):
		child1_gene_from_P1 = []
		child1_gene_from_P2 = [] 
		elements = sorted(random.sample(range(0, self.chromosome_size), 2))
		start_index = elements[0]
		end_index = elements[1]

		for i in range(start_index, end_index):
			child1_gene_from_P1.append(parent1[i])
			
		child1_gene_from_P2 = [item for item in parent2 if item not in child1_gene_from_P1]

		child1 = child1_gene_from_P1 + child1_gene_from_P2
		child1.remove(self.origin_index)
		child1 = [self.origin_index] + child1
		return child1

	def reproduce(self,parent1,parent2):
		'''
			Reproduce using parent1 and parent2 and a crossover
			 strategy.
		'''
		child1 = self.crossover(parent1,parent2)
		child2 = self.crossover(parent1,parent2)

		# elements = sorted(random.sample(range(0, self.chromosome_size), 2))
		# start_index = elements[0]
		# end_index = elements[1]

		# for i in range(start_index, end_index):
		# 	child2_gene_from_P2.append(parent2[i])
				
		# child2_gene_from_P1 = [item for item in parent1 if item not in child2_gene_from_P2]

		# child2 = child2_gene_from_P2 + child2_gene_from_P1
		# child2.remove(self.origin_index)
		# child2 = [self.origin_index] + child2

		return child1,child2


	def mutate1(self,child):
		'''
			Mutation Strategy
		'''
		# TODO: Implement a mutation strategy
		for item in range(len(child)):
			if(random.random() < self.mutation_rate):
				temp = int(random.random() * len(child))
				
				point1 = child[item]
				point2 = child[temp]
				child[item] = point2
				child[temp] = point1

		child.remove(self.origin_index)
		child = [self.origin_index] + child
		return child

	def mutate(self,child):
		'''
			Mutation Strategy
		'''
		# TODO: Implement a mutation strategy
		if(random.random() < self.mutation_rate):
			random.shuffle(child)
		child.remove(self.origin_index)
		child = [self.origin_index] + child
		return child

	def print_result(self):
		'''
			Displays the resulting route in the console.
		'''
		individual = self.population[0][0]
		fitness = self.population[0][1]

		print(" Final Route in %d Generations" % self.generations)
		print(" Final Distance : %5.3f\n" % -fitness)

		counter = 1

		for index in individual:
			print ("%2d. %s" % (counter, self.cities[index]))
			counter += 1

		print ("%2d. %s" % (counter, self.cities[self.origin_index]))


	def run(self):
		'''
			Run the genetic algorithm. Note that this method initializes the
			 first population.
		'''
		generations = 0

		# TODO: Update Initialization
		self.initialize_population()

		last_fitness = 0
		fitness_counter = 0

		while generations <= self.generations:
			new_population = self.population [:int((self.population_size*0.1))]
			# new_population = []
			while len(new_population) < self.population_size:
				# TODO: Update selection
				parent1, parent2 = self.select_parents()
				# TODO: Update reproduction
				child1, child2 = self.reproduce(parent1,parent2)
				
				# TODO: Update Mutation
				# Generate a random number, and only mutate if the number
				#  is below the mutation rate.
				if (random.random() < self.mutation_rate):
					child1 = self.mutate(child1)
					
				if (random.random() < self.mutation_rate):
					child2 = self.mutate(child2)
					

				fitness1 = self.fitnessfcn(child1)
				fitness2 = self.fitnessfcn(child2)

				new_population.append([child1,fitness1])
				new_population.append([child2,fitness2])

			generations = generations + 1

			# Sort the new population in descending order
			new_population.sort(key=operator.itemgetter(1),reverse=True)

			self.population = new_population

			# TODO: Change display rate as needed. Set by 1000 as default.
			if generations % 10 == 0 or generations >= self.generations:
				print("Generation: %d" % generations,"Fitness: %f" % self.population[0][1])
				if generations == self.generations:
					plotTSP(generations, self.population[0][0], self.points, self.population[0][1],True)
				else:
					plotTSP(generations, self.population[0][0], self.points, self.population[0][1],False)

			self.values.append(self.population[0][1])

		self.print_result()


if __name__ == '__main__':

	po_coordinates = "F:/2-2/AI/Project2/project2/coordinates.txt"
	post_office_names = "F:/2-2/AI/Project2/project2/post_offices.txt"
	start_office = "Johnson City Post Office, TN"
	locations = list(np.loadtxt(po_coordinates,delimiter=','))
	cities = [line.rstrip('\n') for line in open(post_office_names)]
	points = []
	paths = []
	print(po_coordinates)
	len(cities)
	start_office_index = [i for i in range(len(cities)) if cities[i] == start_office][0]

	loc_x = [x for x,y in locations]
	loc_y = [y for x,y in locations]
	loc_c = ["black" for _ in range(len(locations))]

	for i in range(0, len(loc_x)):
		points.append((loc_x[i], loc_y[i]))

	#origin, generations, points, population_size, mutation_rate
	origin = (locations[start_office_index][0],locations[start_office_index][1])

	# TODO: Adjust parameters as needed
	# Parameters: 1. origin location,
	#             2. number of generations,
	#             3. locations as a list of tuples,
	#             4. list of city names,
	#             5. number of individuals in each generation,
	#             6. mutation rate

	# gs = GeneticSearch(origin, 230, points, cities, 300, 0.15) -6.79
	# gs = GeneticSearch(origin, 230, points, cities, 350, 0.15) -7.57
	# gs = GeneticSearch(origin, 230, points, cities, 400, 0.15) -7.47
	# gs = GeneticSearch(origin, 150, points, cities, 800, 0.10) #6.5
	# gs = GeneticSearch(origin, 150, points, cities, 1500, 0.3)  -6.44

	gs = GeneticSearch(origin, 150, points, cities, 2200, 0.15)  #  6.076



	gs.run()

	# x = input("Press Enter to Exit...")
	plt.close()
