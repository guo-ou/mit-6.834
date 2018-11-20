import execute
import visualization
import unittest
from semantic_maps import *
from geo_utils import *

'''
This file contains tests for our different functions
'''

def green_test_message():
	try:
		from IPython.display import display_html
		display_html("""<div class="alert alert-success"><strong>Tests passed!!</strong></div>""", raw=True)
	except:
		print ("Tests passed!!")

def test_particle_filter(update_function):
	world, path = execute.load_map_from_json("lake.json")
	robot = Robot(world, 1200)
	localizer = robot.localizer
	localizer.update = lambda *args, **kwargs: update_function(localizer, *args, **kwargs)
	cmd = execute.path_to_command(path, robot, resample_every = 4)
	visualization.simulate(cmd, world, localizer)
	#after window is closed
	exp_loc = localizer.expected_location()
	dist_err = world.robot_pose.distance_to(exp_loc.x, exp_loc.y)
	if dist_err > 50:
		raise AssertionError("Expected location must be 50 away from actual when window closed. Expected distance was " + str(round(dist_err, 1)))
	print ("Actual location  :  ( ", round(world.robot_pose.x, 1), ",", round(world.robot_pose.y, 1) ,")")
	print ("Expected location:  ( ", round(exp_loc.x, 1), ",", round(exp_loc.y, 1) ,")")
	print ("Distance error: ", round(dist_err, 1))
	green_test_message()

def test_localization(map_file_name, update_function, num_particles = 1200, sense_every = 4, resample_every = 6, sigma_forward = 3., sigma_angle = math.radians(5) ):
	world, path = execute.load_map_from_json(map_file_name)
	robot = Robot(world, num_particles)
	localizer = robot.localizer
	localizer.sigma_forward = sigma_forward
	localizer.sigma_angle = sigma_angle
	localizer.update = lambda *args, **kwargs: update_function(localizer, *args, **kwargs)
	cmd = execute.path_to_command(path, robot, sense_every = sense_every, resample_every = resample_every)
	visualization.simulate(cmd, world, localizer)
	#after window is closed
	exp_loc = localizer.expected_location()
	dist_err = world.robot_pose.distance_to(exp_loc.x, exp_loc.y)
	print ("Actual location  :  ( ", round(world.robot_pose.x, 1), ",", round(world.robot_pose.y, 1) ,")")
	print ("Expected location:  ( ", round(exp_loc.x, 1), ",", round(exp_loc.y, 1) ,")")
	print ("Distance error: ", round(dist_err, 1))
	green_test_message()

def test_my_map(map_file_name):
	world, path = execute.load_map_from_json(map_file_name)
	if not(type(path) == type(list()) and len(path) > 0):
		raise ValueError("Path must be at least length 1")
	fig, ax = plt.subplots()
	ax.set_xlim([0,world.width])
	ax.set_ylim([0,world.height])
	visualization.draw_landmarks(ax, world.landmarks)
	visualization.draw_path(path)
	Q_robot = visualization.draw_robot(world.robot_pose)
	plt.show()
	green_test_message()

