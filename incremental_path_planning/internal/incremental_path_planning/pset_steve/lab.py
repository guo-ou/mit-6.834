# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:00:37 2018

@author: GuoOu
"""

from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from tests import *
from world import World
from grid import *
from prioqueue import *
from graph import get_intended_path

inf = float("inf")

# Example: creating a grid from string and drawing it. We can easily create grids from strings
#  by using the Grid.create_from_str method. The grid string should be 'rectangular' with
#  entries as defined below, delimited by spaces:
#   0 -> free cell
#   1 -> obstacle
#   S -> starting point
#   G -> the goal
grid_str = """0 0 0 0 S
              0 1 1 1 0
              0 1 1 1 1
              0 1 0 1 G
              0 0 0 0 0"""
grid = Grid.create_from_str(grid_str)
d = grid.draw()
#In the drawing above, black cells represent obstacles, white cells are passable, the start node is green, and the goal node is red.

#Convert your grid to a graph
#We will not be using this visualization much, since it's hard to tell from a quick glance what is going on.
#graph = grid.to_graph()
#d = graph.draw()

# An example of viewing and updating the world, drawing the
# world state at each time step.

# The World constructor takes in the grid, as well as the robot's start position as a tuple.
#world = World(grid, (4, 4))
#d = world.draw()

#Observe the ground truth, which represents the true environment that the robot doesn't necessarily "know" about.
#d = world.draw(ground_truth=True)

# Move the robot and observe how its knowledge of the world changes:
# Note that the input to update_world is an intended path, specified
# as a list of nodes that the robot plans to traverse to reach its goal.
# Recall that the robot started at (4,4). Hence, the first element of
# the path argument is *not* (4,4), but is instead the *next* step for
# the robot.
world = World(grid, (4, 4))
#world.update_world([(3,3),(3,2),(3,1),(4,0)])
d = world.draw()

# Start with a fresh world with the robot at start position (4,4), and
#  call update_world several times, each time with the new "intended path"
#  for the robot. Once, the robot reaches (4,0), we call world.draw_all_path()
#world = World(grid, (4, 4))
#world.update_world([(3,3),(3,2),(3,1),(4,0)])
#world.update_world([(3,2),(3,1),(4,0)])
#world.update_world([(3,1),(4,0)])
#world.update_world([(4,0)])
#
#world.draw_all_path()
#d = world.draw()

#D* implementation
"""
Our implementation of the algorithm uses the following variables:
    g: a dictionary mapping nodes to their current g-values
    rhs: a dictionary mapping nodes to their current rhs-values
    key_modifier: a number representing the current key modifier k_m
    queue: a PriorityQueue using calc_key as its priority function
    graph: the Graph representing the robot's initial map of the world. You'll want to update graph whenever the world changes.
"""
def grid_heuristic(node1, node2):
    """
    Given two nodes as (x,y) grid-coordinate tuples (e.g. (2,3)), computes the
    heuristic grid-based heuristic value between the nodes.
    (Hint: The heuristic value is just the maximum of the difference in x or y.)
    
    >>>grid_heuristic((1,6), (3,6))
    2
    """
    # YOUR CODE HERE
    return max(abs(node1[0] - node2[0]), abs(node1[1] - node2[1]))

def calc_key_helper(node, g, rhs, start, key_modifier, heuristic=grid_heuristic):
    """
    Computes the node's current key and returns it as a tuple of two numbers.
    
    >>>calc_key_helper((1,1), {(1,1):20}, {(1,1):30}, (0,3), 100)
    (122, 20)
    
    >>>calc_key_helper((1,1), {(1,1):30}, {(1,1):20}, (0,3), 100)
    (122,20)
    
    >>>calc_key_helper((0,3), {(0,3):2, (1,0):200, (0,1):1, "garbage":"nonsense"}, {(0,3):99, (600,600):7}, (600,600), 40)
    (642,2)
    
    >>>calc_key_helper("nodeA", {"nodeA":21}, {"nodeA":33}, "S", 500, zero_heuristic)
    (521,21)
    """
    # YOUR CODE HERE
    val2 = min(g[node], rhs[node])
    val1 = key_modifier + heuristic(start, node) + val2
#    print(val1, ', ',val2)
    return (val1, val2)

#test_grid_heuristic(grid_heuristic)
#test_calc_key_helper(calc_key_helper)
#test_ok()


def update_vertex_helper(node, g, rhs, goal, graph, queue):
    """
    As in the D* Lite pseudocode, this method updates node's rhs value and
    queue status. Returns nothing.
    """
    # YOUR CODE HERE
#    print('\nnode ', node, ', g ', g, ', rhs ', rhs, ', goal ', goal, ', graph ', graph)
    val = 0
    for succ in graph.get_successors(node):
        val += graph.get_edge_weight(node, succ)
    rhs[node] = val
    queue.remove(node)
    
test_update_vertex_helper(update_vertex_helper)
test_ok()

#def compute_shortest_path_helper(g, rhs, start, goal, key_modifier, graph, queue):
#    """As in the D* Lite pseudocode, this method computes (or recomputes) the
#    shortest path by popping nodes off the queue, updating their g and rhs
#    values, and calling update_vertex on their neighbors.  Returns nothing."""
#    # Helper functions that take in only one argument, node:
#    def calc_key(node):
#        return calc_key_helper(node, g, rhs, start, key_modifier)
#    def update_vertex(node):
#        return update_vertex_helper(node, g, rhs, goal, graph, queue)
#    
#    # YOUR CODE HERE
#    raise NotImplementedError
#
#
#test_compute_shortest_path_helper(compute_shortest_path_helper,calc_key_helper)
#test_ok()
#
#
#def dstar_lite(problem):
#    """Performs D* Lite to incrementally find a shortest path as the robot
#    moves through the graph.  Updates the IncrementalSearchProblem, problem, by
#    calling problem.update_world.  The search terminates when the robot either
#    reaches the goal, or finds that there is no path to the goal.  Returns the
#    modified problem.
#
#    Note: The world is dynamic, so the true positions of obstacles may change as
#    the robot moves through the world.  However, if the robot determines at any
#    point that there is no finite path to the goal, it should stop searching and
#    give up, rather than waiting and hoping that the world will improve.
#    """
#
#    ############################################################################
#    # INITIALIZE
#
#    # Get the start node, goal node, and graph from the IncrementalSearchProblem
#    start = problem.start_node
#    goal = problem.goal_node
#    graph = problem.get_graph()
#
#    # Set g=inf and rhs=inf for all nodes, except the goal node, which has rhs=0
#    g = {node:inf for node in graph.get_all_nodes()}
#    rhs = {node:inf for node in graph.get_all_nodes()}
#    rhs[goal] = 0
#
#    # Set the key modifier k_m to 0
#    key_modifier = 0
#
#    # Define shortened helper functions
#    def calc_key(node):
#        return calc_key_helper(node, g, rhs, start, key_modifier)
#    queue = None # to be reinitialized later
#    def update_vertex(node):
#        return update_vertex_helper(node, g, rhs, goal, graph, queue)
#    def compute_shortest_path():
#        return compute_shortest_path_helper(g, rhs, start, goal, key_modifier, graph, queue)
#    heuristic = grid_heuristic
#
#    # Initialize the queue using the priority function calc_key
#    queue = PriorityQueue(f=lambda node: calc_key(node))
#    queue.insert(goal)
#
#    ############################################################################
#    # YOUR CODE HERE
#    raise NotImplementedError
#    
#    return problem
#
## This test uses the example from page 6 of Koenig & Likhachev's paper, referenced above.
#test_dstar_lite(dstar_lite)
#test_ok()
#
#hard_problem_done = run_dstar_lite_hard_grid(dstar_lite)
#hard_problem_done.draw_all_path(time_step=0.5)

