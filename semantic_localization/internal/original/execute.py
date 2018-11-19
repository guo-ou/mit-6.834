import math
import numpy as np
import json


from geo_utils import Pose
from shapely.geometry import Point, Polygon
from semantic_maps import *
from visualization import simulate

BLANK_MAP = {}
BLANK_MAP['robot_pose'] = {}

def path_to_command(path, robot, sense_every = 3, resample_every = 6, loop = True):
    '''
    given a list of points, makes a function that tells the robot to follow the path
    '''
    def command(num, robot):
      eps_dist = 10
      if robot.path == None:
          robot.path = 0
      elif robot.path < len(path):
          x, y = path[robot.path]
          if robot.world.robot_pose.distance_to(x, y) < eps_dist:
              robot.path += 1
              if loop:
                robot.path %= len(path)
          if robot.path < len(path):
              x, y = path[robot.path]
              bearing = robot.world.robot_pose.bearing_to(x, y)
              
              robot.move(10, bearing)
      if num % sense_every == sense_every - 1:
          robot.sense(False)
      if num % resample_every == resample_every - 1:
          robot.localizer.resample()
          robot.localizer.normalize_poses()
      
    return lambda num: command(num, robot)

def load_map_from_json(file_name):
    with open(file_name, "r") as f:
        map_dict = json.load(f)
        landmarks = []
        for class_type, vertices in map_dict['landmarks']:
            landmarks += [Landmark(class_type, vertices)]
        robot_pose = Pose(**map_dict['robot_pose'])
        if 'path' in map_dict:
            path = map_dict['path']
        else:
            path = []
    return (Map(map_dict['width'], map_dict['height'], landmarks, robot_pose), path)


def write_to_json(file_name, obj):
    '''
    obj dictionary must contain:
    'width' : width of map
    'height': height of map
    'landmarks': list of (class_type, [(x, y), ... (x, y)]) tuples
    'robot_pose':
        'x': x position
        'y': y position
        'theta': angle
    'path': list of (x, y) locations 
    '''
    with open(file_name, "w") as f:
        f.write(json.dumps(obj, indent= 4))
   
def main():
    world, path = load_map_from_json("lake.json")
    robot = Robot(world, 1200)
    robot.path = None
    localizer = robot.localizer

    cmd = path_to_command(path, robot)

    simulate(cmd, world, localizer)
    print "DONE"

if __name__ == "__main__": 
    main()
