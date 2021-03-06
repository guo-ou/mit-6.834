import time
import random
from tests import *
from game import Node
from sim import *

###########################################################
# monte carlo tree search algorithm using UCT heuristic
# Input: class Board represents the current game board
#        time limit of calculation in second
# Output: class Action represents the best action to take
##########################################################

def uct(board, time_limit):
    # record start time
    start_time = time.time()
    root = Node(board, None, None)
    c = 5
    
    while (time.time() - start_time) < time_limit:
        tree_terminal = tree_policy(root, c)
        reward_vector = default_policy(tree_terminal.get_board())
        backup(tree_terminal, reward_vector)
        
    return best_child(root, 0).get_action()

#test_uct(uct)

###########################################################
# heuristically search to the leaf level
# Input: a node that want to search down and the
#        exploitation value c
# Output: the leaf node that we expand till
##########################################################
def tree_policy(node, c):
    while not node.get_board().is_terminal():
        if not node.is_fully_expanded():
            return expand(node)
        node = best_child(node, c)
    return node

#test_tree_policy(tree_policy, expand, best_child)

###########################################################
# expand a node since it is not fully expanded
# Input: a node that want to be expanded
# Output: the child node
##########################################################
def expand(node):
    # get the current board
    board = node.get_board()
    
    visited_actions = set([child.get_action() for child in node.get_children()])
    all_actions = board.get_legal_actions()
    
    # get first unvisited
    for all_action in all_actions:
        not_visited = True
        for visited_action in visited_actions:
            if hash(visited_action) == hash(all_action):
                not_visited = False
        if not_visited:
            action = all_action
            break
    
    new_board = action.apply(board)
    child = Node(new_board, action, node)
    node.add_child(child)
    return child

#test_expand(expand)

###########################################################
# get the best child from this node (using heuristic)
# Input: a node, which we want to find the best child of
#        c, the exploitation constant
# Output: the child node
###########################################################
def best_child(node, c):
    children = node.get_children()
    pairs = [(child, child.value(c)) for child in children]
    child, _ = max(pairs, key=lambda p: p[1])
    return child

#test_best_child(best_child)

#######################################################################
# randomly picking moves to reach the end game
# Input: BOARD, the board that we want to start randomly picking moves
# Output: the reward vector when the game terminates
#######################################################################
def default_policy(board):
    while not board.is_terminal():
        actions = board.get_legal_actions()
        action = random.choice(list(actions))
        board = action.apply(board)
    return board.reward_vector()

#test_default_policy(default_policy)

###########################################################
# reward update for the tree after one simulation
# Input: a node that we want to backup from
#        reward value
# Output: nothing
###########################################################
def backup(node, reward_vector):
    while node.get_parent() is not None:
        node.visit()
        node.q += reward_vector[node.get_parent().get_player_id()]
        node = node.get_parent()
    node.visit()

#test_backup(backup)

if __name__=="__main__":
    run_final_test(uct, 2)