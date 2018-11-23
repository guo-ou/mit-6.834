import IPython
from nose.tools import assert_equal, ok_
from classes import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")

def test_get_reachable(get_reachable_by_step):
    var = str(get_reachable_by_step([state(0,0)],3))
    print(var)
    result =  var == "[S(-2,3), S(1,0), S(0,-3), S(-2,1), S(1,2), S(-1,-1), S(1,-2), S(2,2), S(2,0), S(0,0), S(0,6), S(-2,-1), S(0,8), S(0,-2), S(1,4), S(-1,3), S(1,6), S(-1,5), S(0,4), S(1,1), S(0,-1), S(-2,2), S(-1,-2), S(1,3), S(-1,1), S(-2,0), S(-1,0), S(1,5), S(0,2), S(2,3), S(2,1), S(2,-1), S(0,5), S(0,3), S(0,7), S(0,9), S(-1,2), S(3,0), S(0,1), S(1,-1), S(-1,4), S(-3,0), S(-1,6)]"
    if not result:
        raise Exception("Does not pass test!")
        

def test_get_reach(get_reach_by_step):
    result =  str( get_reach_by_step([state(0,0)],3)) == "[S(-2,3), S(1,0), S(0,-3), S(-2,1), S(1,2), S(-1,1), S(1,-2), S(2,2), S(0,6), S(-2,-1), S(0,8), S(-1,3), S(1,4), S(0,0), S(1,6), S(-1,5), S(0,4), S(1,1), S(-1,0), S(-2,2), S(-1,-2), S(1,3), S(0,-1), S(1,5), S(0,2), S(2,3), S(2,1), S(2,-1), S(0,5), S(0,3), S(0,7), S(0,9), S(-1,2), S(3,0), S(0,1), S(-1,4), S(-3,0), S(-1,6)]"
    if not result:
        raise Exception("Does not pass test!")

def test_intersection(get_intersection, states):
    obstacles = [obstacle("tree",[state(1,0)]),obstacle("tree",[state(5,5)])]
    result = str( get_intersection(states,obstacles) ) == "[S(1,0)]"
    if not result:
        raise Exception("Does not pass test!")

def graph_states(states):
    M=10
    N=10
    zvals = [ [0]*M for _ in range(N) ] #np.random.rand(10,10)*10-5

    for my_state in states:
        if my_state.x >= 0 and my_state.y >= 0 and my_state.x < 10 and my_state.y < 10:
            zvals[my_state.y][my_state.x] = -5
            
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['blue','white','green'])
    bounds=[-6,-2,2,6]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    img = plt.imshow(zvals,interpolation='nearest',
                        cmap = cmap,norm=norm,origin="lower")

    plt.grid(True,color='black')
    plt.show()
    
def graph_state_interesections(get_intersection,states,obstacles):
    M=10
    N=10
    zvals = [ [0]*M for _ in range(N) ] #np.random.rand(10,10)*10-5

    for my_state in states:
        if my_state.x >= 0 and my_state.y >= 0 and my_state.x < 10 and my_state.y < 10:
            zvals[my_state.y][my_state.x] = 1
    for obstacle in obstacles:
        for my_state in obstacle.states:
            if my_state.x >= 0 and my_state.y >= 0 and my_state.x < 10 and my_state.y < 10:
                if zvals[my_state.y][my_state.x] == 1:
                    zvals[my_state.y][my_state.x] = 3
                else:
                    zvals[my_state.y][my_state.x] = 2
            
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['white','blue','green','red'])
    #bounds=[-6,-2,2,6]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=False)
    
    img = plt.imshow(zvals,interpolation='nearest',
                        cmap = cmap, norm = norm, origin="lower")

    plt.grid(True,color='black')
    plt.show()
    
    
def draw_init_map():
    M=10
    N=10
    zvals = [ [0]*M for _ in range(N) ] #np.random.rand(10,10)*10-5

    zvals[2][2] = 1
    
    zvals[8][8] = 2
    zvals[9][8] = 2
    zvals[8][9] = 2
    zvals[9][9] = 2
    
    zvals[1][0] = 2
    zvals[1][1] = 2
    
    zvals[4][4] = 2
    zvals[4][5] = 2
    zvals[5][5] = 2
            
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['white','blue','green','red'])
    #bounds=[-6,-2,2,6]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=False)
    
    img = plt.imshow(zvals,interpolation='nearest',
                        cmap = cmap, norm = norm, origin="lower")

    plt.grid(True,color='black')
    plt.show()
    