import time
import numpy as np
import random
from operator import itemgetter
from copy import deepcopy

# ==============================================================================
# Goal: Solve the 8-puzzle using A*Search
# ==============================================================================

#===============================================================================
# Method: method_timing
#  Purpose: A timing function that wraps the called method with timing code.
#     Uses: time.time(), used to determine the time before an after a call to
#            func, and then returns the difference.
def method_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print ('%s took %0.3f ms' % (func, (t2-t1)*1000.0))
        return [res,(t2-t1)*1000.0]
    return wrapper

#===============================================================================
# Class: PriorityQueue
#  Purpose: A simplified PriorityQueue
class PriorityQueue:


	def __init__(self):
		self.queue = []


	def put(self, item, priority):
		node = [item,priority]
		self.queue.append(node)
		self.queue.sort(key=itemgetter(1))


	def get(self):
		if len(self.queue) == 0:
			return None
		node = self.queue.pop(0)
		return node[0]


	def empty(self):
		return len(self.queue) == 0


#===============================================================================
# Class: Board
#  Purpose: Represents an 8-puzzle game Board
class Board:


    def __init__(self):
        '''
            Default Constructor
        '''
        self.board = random.sample(range(0,9),9)
        while not self.solvable():
            self.board = random.sample(range(0,9),9)


    def __init__(self, board):
        '''
            Default Constructor
        '''
        self.board = board
        if not self.solvable():
            raise Exception("This Board is not Solvable.")


    def solvable(self):
        '''
            Returns True if the board is solvable (inversions must be even)
        '''
        return self.inversions()%2 != 1


    def inversions(self):
        '''
            Count the inversions in the board
        '''
        inv = 0
        numbers = deepcopy(self.board)
        numbers.remove(0)

        inv = len([ (i,j) for i in range(len(numbers)) for j in range(i+1,len(numbers)) if(numbers[j]>numbers[i]) ])

        return inv


    def to_s(self):
        '''
            Returns a string representation of the Board
        '''
        shaped = np.reshape(np.array(self.board),(3,3))
        string = '\n'.join(' '.join(map(str, x)) for x in shaped)
        return string.replace("0"," ")


    def to_s(self,state):
        '''
            Returns a string representation of the state passed
        '''
        shaped = np.reshape(np.array(state),(3,3))
        string = '\n'.join(' '.join(map(str, x)) for x in shaped)
        return string.replace("0"," ")



#===============================================================================
# Class: Solver
#  Purpose: Solves an 8-puzzle Board
class Solver:


    def __init__(self,board):
        '''
            Constructor: accepts a game board
        '''
        self.board = board
        self.solution = [1,2,3,4,5,6,7,8,0]


        self.rules = [[] for x in range(9)]
        self.rules[0] = [1,3]
        self.rules[1] = [0,2,4]
        self.rules[2] = [1,5]
        self.rules[3] = [0,4,6]
        self.rules[4] = [1,3,5,7]
        self.rules[5] = [2,4,8]
        self.rules[6] = [3,7]
        self.rules[7] = [4,6,8]
        self.rules[8] = [5,7]

        
    def hamming(self,state):
        '''
            Hamming heuristic: Determine the wrong positions for a given state
        '''
        errors = len([ x for x in range(9) if state[x] != self.solution[x] ])
        return errors


    def manhattan(self,state):
        '''
            Manhattan Heuristic: Determine the Sum of the Manhattan distances for a given state
        '''
        man_dist = 0
        for i,item in enumerate(state):
            curr_row,curr_col = int(i/ 3) , i % 3
            goal_row,goal_col = int(item /3),item % 3
            man_dist += abs(curr_row-goal_row) + abs(curr_col - goal_col)

        return man_dist


    def swap(self,state,a,b):
        '''
            Swap state a and state b
        '''
        state[a], state[b] = state[b], state[a]
        return state


    def get_neighbors(self,state):
        '''
            Returns a list of valid neighbors for the given position
              State is a list of nine numbers.
        '''
        neighbors = []

        for loc in range(9):
            if state[loc] == 0:
                break

        for swap_index in self.rules[loc]:
            neighbor = deepcopy(state)
            neighbor = self.swap(neighbor,loc,swap_index)
            neighbors.append(neighbor)

        return neighbors


    @method_timing
    def astar_search(self):
        '''
            Performs the A* search for the current board
        '''
        steps = 0                       # Initialize the step counter
        board = self.board.board        # Copy of the board

        frontier = PriorityQueue()      # Create the frontier as a PriorityQueue
        frontier.put(board,0)           # - And push the initial state to the frontier

        came_from = {}                  # Use a dictionary to track the path
        cost_so_far = {}                # Use a dictionary to track the path cost by state

        came_from[str(board)] = None    # Push the initial state and set it's parent as None
        cost_so_far[str(board)] = 0     # The cost so far from the initial state is 0.

        ## - - - -
        # TODO: Complete the A* Algorithm Here.
        #      Pseudocode for the algorithm is available on Wikipedia (which,
        #       is actually a better pseudocode than the Best-First Search from the book).
        #       https://en.wikipedia.org/wiki/A*_search_algorithm

        while frontier:

            current = frontier.get()
            if current == self.solution:
                break

            for n in self.get_neighbors(current):
                path_cost = cost_so_far[str(current)] + 1
                if str(n) not in cost_so_far or path_cost < cost_so_far[str(n)]:
                    cost_so_far[str(n)] = path_cost
                    print(self.manhattan(current))
                    priority = path_cost + self.manhattan(current)
                    frontier.put(n,priority)
                    came_from[str(n)] = current

        # End code
        ## - - - -
        node = self.solution

        ## TODO: REMOVE THE FOLLOWING LINE OF CODE, OR THE SOLUTION WILL NOT BE DISPLAYED.
        # node = board

        #-- Recreate the solution by walking back from the goal to the initial state
        solution = []

        while node != None:
            solution.append(node)
            parent = came_from[str(node)]
            node = parent

        #-- Reverse the solution so you move forward
        solution.reverse()

        for state in solution:
            print(self.board.to_s(state),"\n")

        print ("\nTotal Moves:",str(len(solution)-1))



#=====================
# Main Algorithm
tester1 = [8,1,3,4,0,2,7,6,5]
tester2 = [1,8,0,4,3,2,5,7,6]
board = Board(tester2)
solver = Solver(board)
solver.astar_search()
