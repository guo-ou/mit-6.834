# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:54:11 2018

@author: GuoOu
"""

from mdp import *
from utils import *
import pdb
import matplotlib.pyplot as plt


# Implement get_reachable_states
def get_reachable_states(s, mdp, policy, reachable_states = None):
    """
    s: start state
    mdp: MDP object
    policy: dictionary mapping state to action
    reachable_states: None or list of reachable states

    returns: list of reachable states from state s given policy
    """
    if not reachable_states:
        reachable_states = [s]
    current = s
    if s in policy:
        action = policy[s]
        for (result_state, prob) in mdp.T(current, action).items():
            if result_state not in reachable_states and prob > 0.0:
                reachable_states.append(result_state)
                get_reachable_states(result_state, mdp, policy, reachable_states)
    return reachable_states

# Implementation of get children function
def get_children(s, mdp):
    """
    s: current_state
    mdp: MDP objects
    
    returns: list of direct children states from state s
    """
    children = []
    for a in mdp.actions(s):
        for (result_state, prob) in mdp.T(s, a).items():
            if result_state not in children and prob > 0.0:
                children.append(result_state)
    return sorted(children)

#Implementation of policy iteration
def policy_evaluation(states, pi, U, R, T, gamma, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    for i in range(k):
        for s in states:
            U[s] = R[s] + gamma * sum([p * U[s] for (p, s1) in T[(s, pi[s])]])
    return U

def simple_heuristic(s, mdp):
    discount_factor = mdp.gamma
    best_goal_cost = 0
    for s in mdp.terminals:
        if mdp.R(s) > best_goal_cost:
            best_goal_cost = mdp.R(s)
    return best_goal_cost/(1-discount_factor)

#Implementation of LAO*
def lao_star(mdp):
    """
    Maximize some cumulative function of the random rewards
    Args:
        mdp: MDP problem to solve
    
    returns: dictionary mapping expanded states to policies
    """
    s_envelope = [mdp.init]
    s_terminal = [mdp.init]
    pi = None
    
    while True:
        r_envelope = {}
        for s in s_envelope:
            if s in s_terminal:
                r_envelope[s] = simple_heuristic(s, mdp)
            else:
                r_envelope[s] = mdp.R(s)
        
        t_envelope = {}
        for s in s_envelope:
            for a in mdp.actlist:
                if s in s_terminal:
                    t_envelope[(s,a)] = {}
                else:
                    t_envelope[(s,a)] = mdp.T(s, a)
        
        partial_mdp = MDP(s_envelope, mdp.actlist, mdp.init, r_envelope, t_envelope, mdp.terminals, mdp.gamma)
                
        # find optimal policy on states in envelope
        pi = policy_iteration(partial_mdp)
        
        # find reachable states
        reachable_states = []
        reachable_states = get_reachable_states(mdp.init, mdp, pi)
        reachable_states = list(set(reachable_states).intersection(s_terminal))
        
        # get children states of reachable states
        reachable_children = []
        for s in reachable_states:
            reachable_children += get_children(s, mdp)
        
        # define new terminal states
        new_terminals = []
        
        # add current terminal states if not in reachable states
        for s in s_terminal:
            if s not in reachable_states:
                new_terminals.append(s)
        
        # add children if not in envelope
        for s in reachable_children:
            if s not in s_envelope:
                new_terminals.append(s)
        
        # add reachable children to envelope
        for s in reachable_children:
            if s not in s_envelope:
                s_envelope.append(s)
                
        # check if intersection between terminal states and reachable states is empty
        if set(s_terminal).isdisjoint(reachable_states):
            return pi
        s_terminal = new_terminals
    
    return pi

if __name__ == "__main__":
    # Visualizing an example graph/grid world
#    mdp = read_mdp("fig17.mdp")
#    grid = mdp.to_grid()
#    grid.draw()
#    
#    print ("Action list:", mdp.actlist)
#    print ("Transition probabilities from (0,1)" )
##    print(mdp.T('(0,1)', '(0,1)'))
#    for (result_state, prob) in mdp.T('(0,1)', '(0,1)').items():
#        print ("Probability quad will land in state: ", result_state, " is ", prob)
    
#    test_get_reachable_states(get_reachable_states)
#    test_get_children(get_children)
#    mdp = read_mdp("fig17.mdp")
#    grid = mdp.to_grid()
#    grid.draw()
#    pi = lao_star(mdp)
##    pi = policy_iteration(mdp)
#    grid.draw_policy(pi)
    
    mdp = read_mdp("fig_17_modified.mdp")
    grid = mdp.to_grid()
    grid.draw()
    pi = lao_star(mdp)
    print(pi)
    grid.draw_policy(pi)

