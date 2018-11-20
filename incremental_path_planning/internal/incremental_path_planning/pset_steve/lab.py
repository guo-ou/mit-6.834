# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:00:37 2018

@author: GuoOu
"""

from __future__ import division
import numpy as np
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
#grid_str = """0 0 0 0 S
#              0 1 1 1 0
#              0 1 1 1 1
#              0 1 0 1 G
#              0 0 0 0 0"""
#grid = Grid.create_from_str(grid_str)
#d = grid.draw()
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
#world = World(grid, (4, 4))
#world.update_world([(3,3),(3,2),(3,1),(4,0)])
#d = world.draw()

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
    return max(map(abs, [node1[i]-node2[i] for i in (0,1)]))

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
    min_g_rhs = min([g[node], rhs[node]])
    return (min_g_rhs + heuristic(start, node) + key_modifier, min_g_rhs)
#
#test_grid_heuristic(grid_heuristic)
#test_calc_key_helper(calc_key_helper)
#test_ok()


def update_vertex_helper(node, g, rhs, goal, graph, queue):
    """
    As in the D* Lite pseudocode, this method updates node's rhs value and
    queue status. Returns nothing.
    """
    # YOUR CODE HERE
    #update rhs value
    if node != goal:
        rhs[node] = min([graph.get_edge_weight(node, neighbor) + g[neighbor]
                         for neighbor in graph.get_successors(node)])
    #update queue status
    if node in queue:
        queue.remove(node)
    if g[node] != rhs[node]:
        queue.insert(node)
    
#test_update_vertex_helper(update_vertex_helper)
#test_ok()

def compute_shortest_path_helper(g, rhs, start, goal, key_modifier, graph, queue):
    """As in the D* Lite pseudocode, this method computes (or recomputes) the
    shortest path by popping nodes off the queue, updating their g and rhs
    values, and calling update_vertex on their neighbors.  Returns nothing."""
    # Helper functions that take in only one argument, node:
    def calc_key(node):
        return calc_key_helper(node, g, rhs, start, key_modifier)
    def update_vertex(node):
        return update_vertex_helper(node, g, rhs, goal, graph, queue)

#    verbose = False #set this to True to enable print statements below
    
#    if verbose: print('> COMPUTE SHORTEST PATH')
    while True:
        smallest_key = queue.top_key()
        if smallest_key >= calc_key(start) and rhs[start] == g[start]:
            return
        node = queue.pop()
#        if verbose: print('> dequeue node', node, 'with h =', grid_heuristic(node, start))
        if smallest_key < calc_key(node):
#            print(smallest_key, calc_key(node), node)
            queue.insert(node)
        elif g[node] > rhs[node]:
            g[node] = rhs[node]
            for next_node in graph.get_predecessors(node):
                update_vertex(next_node)
        else:
            g[node] = inf
            for next_node in graph.get_predecessors(node) + [node]:
                update_vertex(next_node)

#test_compute_shortest_path_helper(compute_shortest_path_helper,calc_key_helper)
#test_ok()

def dstar_lite(problem):
    """Performs D* Lite to incrementally find a shortest path as the robot
    moves through the graph.  Updates the IncrementalSearchProblem, problem, by
    calling problem.update_world.  The search terminates when the robot either
    reaches the goal, or finds that there is no path to the goal.  Returns the
    modified problem.

    Note: The world is dynamic, so the true positions of obstacles may change as
    the robot moves through the world.  However, if the robot determines at any
    point that there is no finite path to the goal, it should stop searching and
    give up, rather than waiting and hoping that the world will improve.
    """

    ############################################################################
    # INITIALIZE

    # Get the start node, goal node, and graph from the IncrementalSearchProblem
    start = problem.start_node
    goal = problem.goal_node
    graph = problem.get_graph()

    # Set g=inf and rhs=inf for all nodes, except the goal node, which has rhs=0
    g = {node:inf for node in graph.get_all_nodes()}
    rhs = {node:inf for node in graph.get_all_nodes()}
    rhs[goal] = 0

    # Set the key modifier k_m to 0
    key_modifier = 0

    # Define shortened helper functions
    def calc_key(node):
        return calc_key_helper(node, g, rhs, start, key_modifier)
    def update_vertex(node):
        return update_vertex_helper(node, g, rhs, goal, graph, queue)
    def compute_shortest_path():
        return compute_shortest_path_helper(g, rhs, start, goal, key_modifier, graph, queue)
    heuristic = grid_heuristic

    # Initialize the queue using the priority function calc_key
    queue = PriorityQueue(f=lambda node: calc_key(node))
    queue.insert(goal)

    ############################################################################
    verbose = False #set this to True to enable print statements below
    
    # Begin algorithm
    last_start = start
    compute_shortest_path()
    print('robot starting at:', start, ' to :', goal)

    while start != goal:
        if g[start] == inf:
            if verbose: print("no path found")
            return problem
        start = min(graph.get_successors(start),
                    key = lambda neighbor: (graph.get_edge_weight(start, neighbor)
                                            + g[neighbor]))
        old_graph = graph.copy()
        if verbose: print(' robot moving to:', start, end='')
        intended_path = get_intended_path(start, goal, graph, g)
#        if verbose: print('intended path:', intended_path)
        graph = problem.update_world(intended_path)
        changed_edges = old_graph.get_changed_edges(graph)
        if changed_edges:
            key_modifier = key_modifier + heuristic(last_start, start)
            last_start = start
            for (old_edge, new_edge) in changed_edges:
                if old_edge and new_edge: #edge simply changed weight
                    update_vertex(old_edge.source)
                elif not old_edge: #new edge was added
                    raise NotImplementedError("Edge addition not yet supported")
                else: #old edge was deleted
                    raise NotImplementedError("Edge deletion not yet supported")
            try:
                compute_shortest_path()
            except:
                print("no path found")
                break
    print('end search')
    return problem #contains path traversed, intended future path, and other info

## This test uses the example from page 6 of Koenig & Likhachev's paper, referenced above.
#test_dstar_lite(dstar_lite)
#test_ok()

def go(file, t=0.1):
    with open(file, "r") as f:
        grid_str_hard = f.read()
    
    grid_hard = Grid.create_from_str(grid_str_hard)
    print(grid_hard.start, ', ', grid_hard.goal)
    world_hard = World(grid_hard, grid_hard.start, vision_radius=3)
    problem_hard = IncrementalSearchProblem(world_hard, grid_hard.start, grid_hard.goal)
    problem = problem_hard.copy()
    dstar_lite(problem)
    problem._world.draw_all_path(t)

if __name__ == '__main__':
#    file = "grids/hard_grid.txt"
#    file = "grids/small_grid.txt"
#    file = "grids/big_grid.txt"
    file = "grids/impossible.txt"
    go(file)
