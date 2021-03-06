from grid import *
from IPython import display
import time as time_library
from tkinter import *

TOLERANCE = 0.01

class World:
    """
    A World object represents the belief-state of a robot. In
    particular, a World object stores what the world (as a grid)
    "looks like" to the robot at any particular point in time, as
    well as storing what the world "really" looks like.

    A World object also stores the current position of the robot as
    a tuple (x,y), as well as the current "time" (time starts at 0 and
    is incremented by 1 whenever update_world is called), and the
    path travelled by the robot thus far in this World.

    The only way to mutate anything in this class is via the update_world
    method.
    """
    def __init__(self, init_grid_ground_truth, robot_start_position, vision_radius=1.5):
        # The radius the robot can "see" for its belief state
        #   1   = see neighboring cells on 4-connected grid
        #   1.5 = see neighboring cells on 8-connected grid
        self._VISION_RADIUS = vision_radius

        # The current time step, robot's position, and path travelled so far
        self._time = 0
        self._robot_position = robot_start_position
        self._path_travelled = [robot_start_position]

        # Stores what the world really looks like currently
        self._grid_ground_truth = init_grid_ground_truth
        self._goal_location = init_grid_ground_truth.get_goal()

        # Boolean flag for whether the robot is at the goal
        self._is_robot_at_goal = (self._robot_position == self._goal_location)

        # Stores what the robot "sees" or "believes" to be the world currently
        self._grid_belief_state = init_grid_ground_truth.clone_template()
        self._init_belief_state()
        self._update_belief_state()

        # Stores history of belief states and ground truths. The 0th element is
        #  the initial state that this World object was constructed with, and
        #  the last element is the current state.
        self._history_belief_state = [self._grid_belief_state.copy()]
        self._history_ground_truth = [self._grid_ground_truth.copy()]

        # Stores history of "future paths" for the robot. These future paths
        #  are ones supplied in each update_world call, e.g. from the D* Lite
        #  algorithm execution.
        self._history_intended_path = [[]]

    def update_world(self, intended_path):
        """
        Updates the ground truth grid AND the belief state grid,
        and returns a GRAPH-version of the belief state.

        Takes argument intended_path, which is the intended path of the robot
        starting at the new robot's position. For example, if the robot's new
        position should be (3, 4), then intended_path should be of the form
        [(3,4), ..., (gx,gy)] where (gx,gy) is the goal state, and the entire
        path represents the path that the algorithm has calculated for the
        robot at this time step.
        """
        if len(intended_path) < 1:
            raise ValueError("Intended path must be of length at least 1.")
        next_robot_position = intended_path[0]

        self._time += 1
        self._path_travelled.append(next_robot_position)

        # Update ground truth, e.g. if there are moving obstacles.
        self._update_ground_truth(intended_path)

        # Store new robot position. Must happen before _update_belief_state
        #  is called.
        self._robot_position = next_robot_position
        self._is_robot_at_goal = (self._robot_position == self._goal_location)

        # Re-calculate belief state. Must happen after the new _robot_position
        #  is set.
        self._update_belief_state()

        # Store the new belief state and ground truth into history
        self._history_belief_state.append(self._grid_belief_state.copy())
        self._history_ground_truth.append(self._grid_ground_truth.copy())

        # Store future path in future path history
        self._history_intended_path.append(intended_path)

        # Graph-ify the belief state and return it
        return self.belief

    def _update_ground_truth(self, intended_path):
        # TODO: MAKE SURE that this method does not put an obstacle at
        #  grid[intended_path[0]]. That is to say, an obstacle should NEVER
        #  be formed on the robot's new (updated) location.
        pass

    def _init_belief_state(self):
        """
        Initializes belief state by forcing it to know the start and goal cells,
        as required by D* Lite.
        """
        truth = self._grid_ground_truth
        start_cells = truth.get_cells_of_type(CELL_START)
        goal_cells = truth.get_cells_of_type(CELL_GOAL)

        for start in start_cells:
            self._grid_belief_state.set_cell(start[0], start[1], CELL_START)
        for goal in goal_cells:
            self._grid_belief_state.set_cell(goal[0], goal[1], CELL_GOAL)

    def _update_belief_state(self):
        """
        Updates the robot's belief state of the world. This
        information typically depends on the robot's position,
        e.g. what the robot can "see" from where it's at.
        """
        num_cols, num_rows = self._grid_ground_truth.size
        rx, ry = self._grid_ground_truth.cell_center(*self._robot_position)
        for col in range(num_cols):
            for row in range(num_rows):
                x, y = self._grid_ground_truth.cell_center(col, row)
                if (rx - x)**2 + (ry - y)**2 <= (self._VISION_RADIUS * (1 + TOLERANCE))**2:
                    self._grid_belief_state.set_cell(col, row, self._grid_ground_truth.get_cell(col,row))
                else:
                    pass
                    # Instead of overwriting e.g. "0"s (free cell) to every cell that
                    #  the robot can't see, just do nothing to them, so it retains its
                    #  "knowledge" of what used to be at the cell, even after it is
                    #  no longer "nearby" that cell. Hence, if that cell has CHANGED,
                    #  the robot doesn't know that.
        return True

    def draw(self, ground_truth=False, at_time=None, animating = False):
        """
        Draws the current grid representation of this World. This drawing includes
        information about obstacles, start/goal nodes, and the robot's current
        position, as well as the robot's path so far and its currently intended
        future path, if applicable.

        To draw the ground_truth world instead of the robot's viewpoint of
        the world, use ground_truth=True.

        To draw the world as it was at some earlier point in time, use
        at_time=t where t is whatever time point you are interested in (0 <= t <= now).
        """
        time = at_time if at_time else self._time
        if time < 0 or time >= len(self._history_belief_state):
            raise ValueError("Invalid time. You supplied time={}. Time must be non-negative and no more than the current time t={}". \
            format(time, self._time))

        grid_list = self._history_ground_truth if ground_truth == True else self._history_belief_state
        grid = grid_list[time]
        future_path = self._history_intended_path[time]
        path_travelled = self._path_travelled[0:time+1]
        robot_position = self._path_travelled[time]

        axes = grid.draw(animating = animating)
        grid.draw_cell_circle(axes, robot_position, color=COLOR["robot"])
        grid.draw_path(axes, future_path, color=COLOR["path-future"], linewidth=2, linestyle="dashed")
        grid.draw_path(axes, path_travelled, color=COLOR["path-travelled"], linewidth=3)

        return axes

    def draw_all_path(self, time_step=1, ground_truth=False):
#        gui = Tk()
#        gui.geometry("800x800")
#        c = Canvas(gui ,width=800 ,height=800)
#        c.pack()
        for time_index in range(0,self.time+1):
            temp = self.draw(at_time=time_index,animating=True, ground_truth=ground_truth)
            fig = temp.get_figure()
            fig.set_size_inches(10,10)
            display.display(fig)
            display.clear_output(wait=True)
#            gui.update()
            time_library.sleep(time_step)
#        gui.mainloop()

    @property
    def robot_position(self):
        return self._robot_position

    @property
    def time(self):
        return self._time

    @property
    def path_travelled(self):
        return self._path_travelled

    @property
    def belief(self):
        """
        D* Lite should call this. It returns the current belief state as a graph.
        """
        return self._grid_belief_state.to_graph()

    @property
    def belief_grid(self):
        """
        Returns current belief state as a grid. If you're only calling this to draw
        the grid, instead consider calling world.draw()
        """
        return self._grid_belief_state

    @property
    def belief_history(self):
        return self._history_belief_state

    @property
    def ground_truth_history(self):
        return self._history_ground_truth

    def __eq__(self, other):
        return (self._VISION_RADIUS == other._VISION_RADIUS
                and self._time == other._time
                and self._robot_position == other._robot_position
                and self._path_travelled == other._path_travelled
                and (self._grid_ground_truth == other._grid_ground_truth).all()
                and self._goal_location == other._goal_location
                and self._is_robot_at_goal == other._is_robot_at_goal
                and (self._grid_belief_state == other._grid_belief_state).all()
                #ignore the next two because they don't like equality checking:
                #  "ValueError: The truth value of an array with more than one
                #   element is ambiguous. Use a.any() or a.all()"
#                and self._history_belief_state == other._history_belief_state
#                and self._history_ground_truth == other._history_ground_truth
                and self._history_intended_path == other._history_intended_path)
