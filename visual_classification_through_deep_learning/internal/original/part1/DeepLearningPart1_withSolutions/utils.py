#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various utiltiy and testing functions
"""
import networkx as nx
import numpy as np
from nose.tools import assert_equal, ok_
from IPython.display import display, HTML, clear_output


from dispatcher import *
import matplotlib.pyplot as plt

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print "Tests passed!!"


def check_offline_scheduler(fn, N=100, epsilon=1e-10):
    stn = create_example_stn_1()
    # Students may (or may not) use randomization in their scheduler algorithm.
    # In case they do, take a Monte-carlo approach and try their scheduler fn
    # a bunch of times to makes sure it works
    for i in xrange(N):
        schedule = fn(stn, 'A')
        satisfied, explanation = check_schedule_against_stn(stn, schedule, epsilon=epsilon)
        if not satisfied:
            raise Exception(explanation)
    return True


def check_online_dispatch(fn, epsilon=0.02):
    stn = create_example_stn_1()
    dispatcher = Dispatcher(sim_time=True, quiet=True)
    try:
        fn(stn, dispatcher)
        result, explanation = check_schedule_against_stn(stn, dispatcher.execution_trace, epsilon=epsilon)
    except Exception as e:
        # If we get an error, return that as explanation
        result, explanation = False, "Online dispatcher threw error: {}".format(e)
    if not result:
        raise Exception(explanation)
    return True


def check_schedule_against_stn(stn, schedule, epsilon=1e-10):
    """ Verifies that the given schedule satisfies all
    of the temporal constraints in the STN.

    Inputs: stn - a networkx.digraph with STC (simple temporal constraints)
                   representing an STN
            schedule - a dictionary, mapping event names (nodes in stn) to
                   float values (scheduled times)

    Output: A tuple (satisfied, explanation). satisfied is True if all
            temporal constraints are satisifed, else its False. explanation is
            a string describing why it's unsatisfied (if it is), otherwise it's
            a kind message.
    """
    for (u, v) in stn.edges():
        # Retrieve the STC for this edge
        lb, ub = stn[u][v]['stc']
        stc_satisfied = (schedule[v] - schedule[u] <= ub + epsilon) and (schedule[v] - schedule[u] >= lb - epsilon)
        if not stc_satisfied:
            return (False, "There's a temporal constraint [{}, {}] from {} to {}, but those events were scheduled at {}={:0.4f} and {}={:0.4f} for a difference of {:0.4f}, violating the temporal constraint!".format(lb, ub, u, v, u, schedule[u], v, schedule[v], schedule[v] - schedule[u]))
    # All edges satisfied
    return (True, "Great!")


def test_distance_graph(dg, stn):
    if set(dg.nodes()) != set(stn.nodes()):
        raise Exception("Distance graph and STN don't have same events!")
    if len(dg.edges()) != 2*len(stn.edges()):
        raise Exception("Wrong number of edges between distance graph and STN!")
    for (u, v) in stn.edges():
        lb, ub = stn[u][v]['stc']
        if dg[u][v]['weight'] != ub:
            raise Exception("Invalid upperbound edge!")
        elif dg[v][u]['weight'] != -lb:
            raise Exception("Invalid lowerbound edge!")
    return True

def test_apsp_example_stn_1(g_apsp):
    # Correct events
    assert_equal(set(g_apsp.nodes()), set(['A', 'B', 'C', 'D']))
    # No self-loops
    for u in g_apsp.nodes():
        ok_(not u in g_apsp[u], msg="Has self-loops but shouldn't!")

    assert_equal(len(g_apsp.edges()), 12, msg="Incorrect number of edges")

    # Correct edges
    assert_equal(g_apsp['A']['B']['weight'], 10.0, msg="Invalid edge weight")
    assert_equal(g_apsp['B']['A']['weight'], -1.0, msg="Invalid edge weight")

    assert_equal(g_apsp['A']['C']['weight'], 9.0, msg="Invalid edge weight")
    assert_equal(g_apsp['C']['A']['weight'], 0.0, msg="Invalid edge weight")

    assert_equal(g_apsp['A']['D']['weight'], 11.0, msg="Invalid edge weight")
    assert_equal(g_apsp['D']['A']['weight'], -2.0, msg="Invalid edge weight")

    assert_equal(g_apsp['B']['C']['weight'], -1.0, msg="Invalid edge weight")
    assert_equal(g_apsp['C']['B']['weight'], 1.0, msg="Invalid edge weight")

    assert_equal(g_apsp['B']['D']['weight'], 1.0, msg="Invalid edge weight")
    assert_equal(g_apsp['D']['B']['weight'], -1.0, msg="Invalid edge weight")

    assert_equal(g_apsp['C']['D']['weight'], 2.0, msg="Invalid edge weight")
    assert_equal(g_apsp['D']['C']['weight'], -2.0, msg="Invalid edge weight")

    return True


def test_minimal_dispatchable_graph_example_1(g_disp):
        # Correct events
        assert_equal(set(g_disp.nodes()), set(['A', 'B', 'C', 'D']))

        assert_equal(len(g_disp.edges()), 6, msg="Incorrect number of edges")

        # Correct edges
        assert_equal(g_disp['A']['C']['weight'], 9.0, msg="Invalid edge weight")
        assert_equal(g_disp['C']['A']['weight'], 0.0, msg="Invalid edge weight")

        assert_equal(g_disp['C']['B']['weight'], 1.0, msg="Invalid edge weight")
        assert_equal(g_disp['B']['C']['weight'], -1.0, msg="Invalid edge weight")

        assert_equal(g_disp['B']['D']['weight'], 1.0, msg="Invalid edge weight")

        # For this example, there are two edges that both lower dominate each other.
        # The students should be able to have both. Namely, edges DB and DC both dominate
        # each other.
        ok_(not(g_disp.has_edge('D', 'B') and g_disp.has_edge('D', 'C')), msg="DC and DB both dominate each other, but your graph has both!")
        if g_disp.has_edge('D', 'B'):
            assert_equal(g_disp['D']['B']['weight'], -1.0, msg="Invalid edge weight")
        else:
            assert_equal(g_disp['D']['C']['weight'], -2.0, msg="Invalid edge weight")

        return True

"""
Example STNs
"""
def create_example_stn_1():
    """ Helper to create and return an example STN."""
    stn = nx.DiGraph()
    stn.add_edge('A', 'B', stc=[0, 10])
    stn.add_edge('B', 'D', stc=[1, 1])
    stn.add_edge('A', 'C', stc=[0, 10])
    stn.add_edge('C', 'D', stc=[2, 2])
    return stn

def create_example_stn_2():
    """ Helper to create and return an example STN."""
    stn = nx.DiGraph()
    stn.add_edge('A', 'B', stc=[1.0, 2.0])
    stn.add_edge('C', 'D', stc=[1.0, 2.0])
    stn.add_edge('E', 'F', stc=[1.0, 2.0])
    stn.add_edge('B', 'C', stc=[0, np.inf])
    stn.add_edge('D', 'E', stc=[0, np.inf])
    stn.add_edge('A', 'F', stc=[0, 4.0])
    return stn

"""
Visualization
"""
def plot_success_ratio_vs_disturbances(online_fn, stn, N_repeats=1000, max_disturbance=1.0):
    """ Perform a Monte-Carlo simulation of the online dispatcher. We
    set up a test example
    """
    noise_vals = np.linspace(0.0, max_disturbance, 20)
    percent_successes = []
    trial = 0.0
    # Iterate over all the desired noise values
    for noise in noise_vals:
        # For each, do a Monte Carlo simulation to estimate the probabilty of a successful execution
        successes = 0
        for i in range(N_repeats):
            # Occasionally update the progress bar
            percent_done = trial / (N_repeats * len(noise_vals))
            if trial % 200 == 0:
                clear_output(wait=True)
                display(HTML(progress_bar_html(percent_done, "Simulating {} online executions...".format(N_repeats * len(noise_vals)))))
            trial += 1.0
            # Set up the dispatcher with the desired noise value, and call the online execution function
            dispatcher = Dispatcher(disturbance_max_delay=noise, quiet=True, sim_time=True)
            try:
                online_fn(stn, dispatcher)
                # Check and see if the execution satisfies all the original problem's temporal constraints, and tally.
                result, explanation = check_schedule_against_stn(stn, dispatcher.execution_trace, epsilon=0.01)
            except Exception as e:
                # If we get an exception, count it as False
                result, explanation = False, "Online dispatcher threw an error: {}".format(e)

            if result:
                successes += 1
        # We've completed this round of Monte Carlo simulation, record it.
        percent_successes.append(1.0 * successes / N_repeats)
    # Finish the progress bar
    clear_output(wait=True)
    display(HTML(progress_bar_html(1.0, "Done!")))
    # Create a plot!
    percent_successes = np.array(percent_successes)
    plt.plot(noise_vals, percent_successes, linewidth=2.0)
    plt.axis([0, max_disturbance, 0.0, 1.1])
    plt.title('Success rate vs Max Timing Disturbance')
    plt.xlabel('Max Timing Disturbance (s)')
    plt.ylabel('Success ratio')
    plt.grid(True)


def progress_bar_html(r, message):
    pct = "{:.2f}".format(100*r)
    return """<div class=\"progress\">
  <div class=\"progress-bar progress-bar-striped progress-bar-success\" role=\"progressbar\" aria-valuenow=\"{}\" aria-valuemin=\"0\" aria-valuemax=\"100\" style=\"width: {}%\">
    {}
  </div>
</div>""".format(pct, pct, message)


def format_num(v):
    if v == np.inf:
        return "∞"
    elif v == -np.inf:
        return "-∞"
    else:
        return str(v)
