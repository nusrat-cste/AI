import math
import numpy as np
import operator
import random
import sys
from matplotlib import pyplot as plot

plot.ion()
plot.figure(figsize=(9,5))

CLUSTERS = 3            # Clusters to attempt
DISPLAY_RATE = 10      # Show a graph at this rate

class GeneticSearch:
    """
        Class: GeneticSearch
    """
    def __init__(self, filename, generations, population_size, mutation_rate):
        '''
            Initialize the GA by reading the points from the file and setting
             standard GA parameters.
        '''
        self.filename = filename
        self.read_file()
        self.population = None
        self.chromosome_size = len(self.points)
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.values = []



    def read_file(self):
        '''
            Read points from the passed file in the format x,y
        '''
        self.points = np.genfromtxt(self.filename, delimiter=',')



    def initialize_population(self):
        '''
            Create the initial population and find the fitness for each
             individual.
        '''
        self.population = []

        for _ in range(self.population_size):
            individual = [random.randint(1,CLUSTERS) for _ in range(self.chromosome_size)]
            fitness = self.fitnessfcn(individual)
            self.population.append([individual,fitness])

        self.population.sort(key=operator.itemgetter(1),reverse=True)



    def sum_squared_distances(self, c_points, center):
        '''
            Calculates the distance from each point in c_points to the center,
             squares it, and adds all of them together.
        '''
        distances = 0

        for pt in c_points:
            distances += (np.linalg.norm(pt-center)**2)

        return distances



    def get_clusters_and_centers(self,individual):
        '''
            Using individual, it returns a dictionary of indexes for points in
            each cluster, a list of points in each cluster, and centers for
            each cluster.
        '''
        clusters = {}
        c_points = {}
        centers = {}
        c_vals = set(individual)

        for key in c_vals:
            clusters[key] = [i for i,x in enumerate(individual) if x == key]
            c_points[key] = [self.points[x] for x in clusters[key]]
            centers[key] = sum(c_points[key])/float(len(c_points[key]))

        return clusters,c_points,centers



    def fitnessfcn(self, individual):
        '''
            The fitness function for the individual is the total
             sum of squared distances for all clusters.
            The lowest value represents the best solution, so this is negated
             to represent "maximum utility" idea of Genetic Algorithms
        '''
        clusters, c_points, centers = self.get_clusters_and_centers(individual)
        sse = {}
        c_vals = set(individual)
        total_sse = 0
        total_bss = 0

        for key in c_vals:
            sse[key] = self.sum_squared_distances(c_points[key],centers[key])

        return -sum(list(sse.values()))

    def reproduce(self,parent1,parent2):
        '''
            Reproduce using parent1 and parent2 and a crossover
             strategy.
        '''
        crossover1 = random.randrange(0,self.chromosome_size)

        ''' Single point crossover:
              Pull bits 0..crossover1 from parentX.
              Pull remaining bits from parentY in the order they appear.
        '''
        child1 = parent2[:crossover1] + parent1[crossover1:]
        child2 = parent1[:crossover1] + parent2[crossover1:]

        value = [child1,child2]

        return value



    def mutate(self,child):
        '''
            Mutation Strategy: Assign three random points to their closest
             cluster center
        '''
        clusters, c_points, centers = self.get_clusters_and_centers(child)
        distance1 = {}
        distance2 = {}
        distance3 = {}

        index1 = random.randint(0,self.chromosome_size-1)
        index2 = random.randint(0,self.chromosome_size-1)
        index3 = random.randint(0,self.chromosome_size-1)

        for key in set(child):
            distance1[key] = np.linalg.norm(self.points[index1]-centers[key])**2
            distance2[key] = np.linalg.norm(self.points[index2]-centers[key])**2
            distance3[key] = np.linalg.norm(self.points[index3]-centers[key])**2

        child[index1] = [key for key,val in distance1.items() if val == min(distance1.values())][0]
        child[index2] = [key for key,val in distance2.items() if val == min(distance2.values())][0]
        child[index3] = [key for key,val in distance3.items() if val == min(distance3.values())][0]

        return child



    def show_step(self, generation, fitness):
        '''
            Plots an intermediate step for the current generation
        '''
        plot.suptitle("Clustering with Genetic Algorithms - Generation " + str(generation) + \
    				 "\nFitness: " + str(fitness))
        best = self.population[0]

        plot.scatter([x for x,y in self.points],[y for x,y in self.points], c=best[0])

        plot.pause(1)



    def show_result(self):
        '''
            Display the final result and the fitness over time
        '''
        plot.suptitle("Clustering with Genetic Algorithms - Generation " + str(self.generations) + \
    				 "\nFitness: " + str(-self.population[0][1]))
        best = self.population[0]

        plot.scatter([x for x,y in self.points],[y for x,y in self.points], c=best[0])

        plot.figure("Genetic Search - Best Fitness by Generation")

        plot.plot(self.values)

        plot.show()
        plot.pause(1)



    def run(self):
        '''
            Run the genetic algorithm. Note that this method initializes the
             first population.
        '''
        self.initialize_population()

        generations = 1

        while generations <= self.generations:
            new_population = []
            parent1 = []
            parent2 = []

            retain = math.ceil(self.population_size*0.05)
            new_population = self.population[:retain]
            while len(new_population) < self.population_size:

                parent1 = []

                parent1 = random.choice(self.population)[0]
                parent2 = random.choice(self.population)[0]

                while parent1 == parent2:
                    parent2 = random.choice(self.population)[0]

                children = self.reproduce(parent1,parent2)

                child1 = children[0]
                child2 = children[1]

                if (random.random() < self.mutation_rate):
                    child1 = self.mutate(child1)
                if (random.random() < self.mutation_rate):
                    child2 = self.mutate(child2)

                fitness1 = self.fitnessfcn(child1)
                fitness2 = self.fitnessfcn(child2)

                new_population.append([child1,fitness1])
                new_population.append([child2,fitness2])

            generations = generations + 1
            new_population.sort(key=operator.itemgetter(1),reverse=True)

            self.population = new_population

            self.values.append(self.population[0][1])

            if generations % DISPLAY_RATE == 0:
                print("Generation",generations,"Fitness",-self.population[0][1])
                self.show_step(generations,-self.population[0][1])




def main():
    # filename = sys.argv[1]  #TODO: Change to the filename 
    # if you do not want to enter this on the command line
    filename = 'F:/2-2/AI/lab3/points2.csv'
    gs = GeneticSearch(filename, 800, 200, 0.5)
    gs.run()
    gs.show_result()

    input("Press Enter to exit...")
    plot.close()

main()
