# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:53:51 2018

@author: GuoOu
"""

from __future__ import division
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import reach_tests as tests
from classes import *

def reachable(start_state):
    x=start_state.x
    y=start_state.y
    reachable = [state(x,y+1),state(x,y+2),state(x,y+3),state(x+1,y),state(x-1,y),state(x,y-1)]
    return reachable
    # raise NotImplementedException("Whoops, forgot to implement this!")

def get_reachable_by_step(states,steps):
    states = states
    for step in range(steps):
        for state in states:
            states = list(set(states + reachable(state)))
    return states

def get_reach_by_step(states,steps):
    old_states = states
    for step in range(steps):
        new_states =[]
        for state in old_states:
            for new_state in reachable(state):
                new_states = list(set(new_states + reachable(state)))
        old_states = new_states
    return new_states


def get_intersection(states,obstacles):
    intersection = []
    for state1 in states:
        for obstacle in obstacles:
            if state1 in obstacle.states:
                intersection.append(state1)
                break
    return intersection

if __name__ == "__main__":
    tests.draw_init_map()
#    
#    # A get-to-know-me state
#    my_state = state(1,2)
#    print (my_state)
#    print (my_state.x)
#    print (my_state.y)
#    
#    # A get-to-know-me obstacle
#    my_ob = obstacle("tree",[state(1,1),state(1,2)])
#    print (my_ob)
#    print (my_ob.name)
#    print (my_ob.states)
#    tests.graph_states(reachable(state(2,2)))
    
#    tests.graph_states(get_reachable_by_step([state(2,2)],2))
#    tests.test_get_reachable(get_reachable_by_step)
    
#    tests.graph_states(get_reach_by_step([state(2,2)],4))
#    tests.test_get_reach(get_reach_by_step)
    
    ob1 = obstacle("tree1",[state(9,9),state(9,8),state(8,8),state(8,9)])
    ob2 = obstacle("tree2",[state(1,0),state(1,1)])
    ob3 = obstacle("tree3",[state(4,4),state(4,5),state(5,5)])
    
    obstacles = [ob1,ob2,ob3]
    
    states = get_reach_by_step([state(0,0)],1)
    tests.graph_state_interesections(get_intersection,states,obstacles)
    tests.test_intersection(get_intersection,get_reach_by_step([state(0,0)],1))
    