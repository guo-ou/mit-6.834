import IPython
import collections
from plan import *

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print "Tests passed!!"

def check_equal(activities1, activities2):
    p1 = Plan(activities1)
    p2 = Plan(activities2)
    assert p1 == p2

universal_goals = ['(collect-goal-passenger-at p5 n5)', '(collect-goal-passenger-at p1 n1)', '(collect-goal-passenger-at p3 n0)', '(collect-goal-passenger-at p2 n7)', '(collect-goal-passenger-at p6 n5)', '(collect-goal-passenger-at p4 n0)', '(collect-goal-passenger-at p0 n8)']
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def check_elevator_always(activities):
    p = Plan(activities)
    collected_goal = p.extract("collect-goal-passenger-at")
    move_up_fast = p.extract("move-up-fast")
    move_down_slow = p.extract("move-down-fast")
    

    assert len(collected_goal) == len(universal_goals)
    assert compare(collected_goal, universal_goals)
    assert not move_up_fast
    assert not move_down_slow
    
def check_sometime_next(activities):
    p = Plan(activities)
    collected_goal = p.extract("collect-goal-passenger-at")
    
    assert len(collected_goal) == len(universal_goals)
    assert compare(collected_goal, universal_goals)
    # Mr.Duck must arrive at the golf club (`n8`) **sometime after** Mrs.Duck arrives at her destination
    assert collected_goal.index("(collect-goal-passenger-at p0 n8)") > collected_goal.index("(collect-goal-passenger-at p1 n1)")
    # Dewey would like to be at his destination right after his brother
    assert collected_goal.index("(collect-goal-passenger-at p4 n0)") -  collected_goal.index("(collect-goal-passenger-at p3 n0)") == 1

def check_recharge(activities):
    duratives = ["move-up-slow", "move-down-slow", "move-up-fast", "move-down-fast", "board", "leave", "recharge"]

    p = Plan(activities)
    collected_goal = p.extract("collect-goal-passenger-at")
    
    assert len(collected_goal) == len(universal_goals)
    assert compare(collected_goal, universal_goals)

    seq = p.extract(*duratives)
    exhausion = { "slow0-0": 0, "slow1-0": 0, "fast0": 0, "fast1": 0 }
    for act in seq:
        for lift in exhausion.keys():
            if lift in act:
                if "recharge" in act:
                    exhausion[lift] = 0
                else:
                    exhausion[lift] += 1
            assert exhausion[lift] < 5