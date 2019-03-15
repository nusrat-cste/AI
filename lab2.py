import os
import time
import numpy
from queue import PriorityQueue
from collections import deque

# ==============================================================================
# Goal: Traverse a directory structure using Depth-First and Breadth-First
#        Search Algorithms
# ==============================================================================

#---------------------------------------
# The Path to Search
#  You may need to alter this based on
#   your file location and OS.
path = "F:/2-2/AI/lab2/treepath"

#---------------------------------------
# The Goal Filename
#  This is the name of the files you are
#   attempting to find.
goal = "YAjrbqc.bin"
# goal = "xhtj8.bin"
# goal = "XUB.bin"


#===============================================================================
# Method: matchfinder
#  Purpose: a string match function to check if the file is the target goal file
def matchfinder(x,y):
        match = False
        if x==y:
           match = True     
        return match 

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
        #print ('%s took %0.3f ms' % (func, (t2-t1)*1000.0))
        return [res,(t2-t1)*1000.0]
    return wrapper

#===============================================================================
# Method: expand
#  Purpose: Returns the child nodes of the current node in a list
#     Uses: os.listdir, which returns a Python list of children--directories
#           as well as files.
def expand(path):
 	return(os.listdir(path))

#===============================================================================
# Method: breadthFirst
#  Purpose: Conducts a Breadth-First search of the file structure
#  Returns: The location of the file if it was found, an empty string otherwise.
#     Uses: Wrapped by method_timing method
@method_timing
def breadthFirst(path, goal):
        result = ''
        frontier = deque([path])   
        explored = []
        while (frontier):
                current_node = frontier.popleft()
                if current_node not in explored: 
                        if os.path.isdir(current_node):
                                children = expand(current_node)
                                for child in children:
                                        node = current_node+'/'+child
                                        frontier.append(node)
                                explored.append(current_node)
                        elif os.path.isfile(current_node):
                                directory, filename = os.path.split(current_node)
                                if matchfinder(filename, goal):
                                        result = current_node
        return result


#===============================================================================
# Method: depthFirst
#  Purpose: Conducts a Depth-First search of the file structure
#  Returns: The location of the file if it was found, an empty string otherwise.
#     Uses: Wrapped by method_timing method
@method_timing
def depthFirst(path, goal):
        result = ''
        frontier = deque([path])   
        explored = []
        while (frontier):
                current_node = frontier.pop()
                if current_node not in explored: 
                        if os.path.isdir(current_node):
                                children = expand(current_node)
                                for child in children:
                                        node = current_node+'/'+child
                                        frontier.append(node)
                                explored.append(current_node)
                        elif os.path.isfile(current_node):
                                directory, filename = os.path.split(current_node)
                                if matchfinder(filename, goal):
                                        result = current_node
        return result



#=====================
# TODO: Main Algorithm
#
#  Completing the code above will allow this code to run. Comment or uncomment
#   as necessary, but the final submission should be appear as the original.
bfs = numpy.empty((10))
for x in range(0,10):
    filelocation = breadthFirst(path, goal)
    if filelocation[0] != "":
        print ("BREADTH-FIRST: Found %s in %0.3f ms" % (goal,filelocation[1]))
        bfs[x] = filelocation[1]

dfs = numpy.empty((10))
for x in range(0,10):
    filelocation = depthFirst(path, goal)
    if filelocation[0] != "":
        print ("  DEPTH-FIRST: Found %s in %0.3f ms" % (goal,filelocation[1]))
        dfs[x] = filelocation[1]

print("\n FULL PATH: %s" % filelocation[0])

print ("\nBREADTH-FIRST SEARCH AVERAGE TIME: %0.3f ms" % bfs.mean())
print (" DEPTH-FIRST SEARCH AVERAGE TIME: %0.3f ms" % dfs.mean())

