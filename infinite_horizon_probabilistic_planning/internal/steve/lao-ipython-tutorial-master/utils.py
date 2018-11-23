# Copied in large part from Utils file from 16.413 PSET 1

import IPython
from nose.tools import assert_equal, ok_
from mdp import *

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print ("Tests passed!!")


"""
Function checking code
"""   
def test_get_reachable_states(fn):
    print ("Checking get_reachable_states implementation...")
    # Test one
    print ("Test one...")
    mdp = read_mdp("test.mdp")
    grid = mdp.to_grid()
    policy = {'(0,0)': '(0,1)'}
    states = fn('(0,0)', mdp, policy)
    assert_equal(states, ['(0,0)','(0,1)'])
    # Test two
    print ("Test two...")
    mdp = read_mdp("test_2.mdp")
    grid = mdp.to_grid()
    policy = {'(0,0)': '(0,1)', '(0,1)': '(0,1)'}
    states = fn('(0,0)', mdp, policy)
    assert_equal(states, ['(0,0)','(0,1)','(1,1)'])
    
def test_get_children(fn):
    print ("Checking get_children implementation...")
    mdp = read_mdp("test.mdp")
    grid = mdp.to_grid()
    # Test one
    print ("Test one...")
    states = fn('(0,0)', mdp)
    assert_equal(states, ['(0,0)','(0,1)'])
    # Test two
    print ("Test two...")
    states = fn('(0,1)', mdp)
    assert_equal(states, ['(1,1)'])
    
def check_final_solution(policy):
    print ("Checking final policy...")
    assert_equal(policy, {'(0,0)': '(1,0)', '(3,0)': None, '(1,0)': '(1,0)', '(2,0)': '(1,0)'})
    