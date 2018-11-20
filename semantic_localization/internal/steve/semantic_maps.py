# 16.412/6.834 Cognitive Robotics Advanced lecture: Semantic Localization
'''
This file contains various structures of semantic maps.
Version 1.0. rmata@mit.edu.
Version 2.0. mtraub@mit.edu

We are interested in getting contextual, qualitative (read: human-readable)
information about the surroundings.

We are trying to keep these structure as generic as possible. In general, they
are wrappers of some kind, collecting rather complex objects in a certain way.
The goal is to read out the relationships between the objects at the end- so we
encode a string outreader functionality.

_Version 1.0_
    - application-based (e.g. indoor location is in mind, although )
    - will contain a Simple Ring and a more complex structure, ComplexRing.

_Version 2.0_
    - framework for localization on map with labeled polygons
    - geometric classes: Pose where Pose adds a bearing [0, 2pi]. Point and Polygon
      are from shapely
    - Map class contain information about worlds and functions for interacting with it
    - Robot class contains functions for modifying its state and belief
    - Localizer implements particle filter localization

'''

import math
import numpy
import matplotlib.pyplot as plt
import random
from geo_utils import Pose
from shapely.geometry import Point, Polygon, LineString, MultiPoint
import scipy.stats as stats
import itertools

class Landmark:
    '''
    polygon: polygon object representing the shape
    class_type: string indicating the class type of the landmark, e.g. "tree"
    '''
    def __init__(self, class_type, vertices):
        self.polygon = Polygon(vertices)
        self.class_type = class_type

    def get_polygon(self):
        return self.polygon

    def get_class(self):
        return self.class_type

    # returns tuples, representing points
    def get_vertices(self):
        return self.polygon.exterior.coords

class Detection:
    '''
    class_type: string indicating class of landmark
    dist: distance from pose to detection
    theta: bearing of pose to detection
    '''
    def __init__(self, class_type, dist, theta):
        self.class_type = class_type
        self.dist = dist
        self.theta = theta

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(" + self.class_type + "; " + str(self.dist) + ", " + str(self.theta) + ")"

class Map:
    '''
    width: width of map
    height: height of map
    landmarks: list of Landmark objects in the map
    classes: set of all class types represented in the map
    robot_pose: the robot's position in the map
    '''
    def __init__(self, width = 500, height = 500, landmarks = [], robot_pose = None):
        self.width = width
        self.height = height
        self.landmarks = landmarks
        self.classes = set([l.class_type for l in self.landmarks])
        self.robot_pose = robot_pose 

    def add_robot(self, pose):
        self.robot_pose = pose

    def add_landmark(self, landmark):
        '''
        add a new landmark to the map
        returns nothing
        '''
        self.landmarks += [landmark]
        self.classes = self.classes.add(landmark.class_type)

    def whatCanISee(self, pose = None, view_distance = 700, view_angle = math.radians(110)):
        '''
        pose: Pose object to simulate perception from
            if None, uses self.robot_pose <---------------- IMPORTANT FEATURE
        view_angle: the width of the view sector
        view_distance: the radius of perception
        returns list of Detection objects
        '''
        if pose == None:
            pose = self.robot_pose

        #determine number of sections to break view_area into
        goal_delta_theta = math.radians(27.5)
        num_pieces = min(int(round(view_angle / goal_delta_theta, 0)), 1)
        delta_theta = view_angle / num_pieces
        vertices = [(pose.x, pose.y)]
        for i in range(num_pieces + 1):
           theta = pose.theta - view_angle / 2. + delta_theta * i
           x = pose.x + view_distance * math.cos(theta)
           y = pose.y + view_distance * math.sin(theta)
           vertices += [(x, y)]
        view_area = Polygon(vertices)


        # intersections = [(l.get_class(), view_area.intersection(l.get_polygon())) for l in self.landmarks]
        # observed_objects_tuple = [(class_type, polygon) for class_type, polygon in intersections if not polygon.is_empty]
        # observed_objects_x_y = [(class_type, (o.bounds[0] + o.bounds[2])/2, (o.bounds[1] + o.bounds[3])/2 ) for class_type, o in observed_objects_tuple]


        visible_landmarks = []

        for lm in self.landmarks:
            verts = lm.get_vertices()
            for v in verts:
                r = pose.distance_to(v[0], v[1])
                theta = pose.bearing_to(v[0], v[1])
                if r < view_distance and abs(theta) <= view_angle / 2.0:
                    # create line from robot to v
                    line = LineString([(pose.x, pose.y), (v[0], v[1])])
                    obstructed = False
                    for other_lm in self.landmarks:
                        # check for intersection
                        if line.crosses(other_lm.get_polygon()):
                            obstructed = True
                            break
                    if not obstructed:
                        visible_landmarks += [lm]
                        break
                


        observed_objects_tuple = [(lm.class_type, view_area.intersection(lm.polygon)) for lm in visible_landmarks]
        observed_objects_x_y = [(o[0], (o[1].bounds[0] + o[1].bounds[2])/2, (o[1].bounds[1] + o[1].bounds[3])/2 ) for o in observed_objects_tuple if len(o[1].bounds) > 0]
        # turn those objects into Detection objects
        observed_objects = [Detection(c, pose.distance_to(x, y), pose.bearing_to(x, y)) for c, x, y in observed_objects_x_y]

        return observed_objects

    def isOccupied(self, loc):
        '''
        loc: Point object to test if is allowed position on the map
        returns True if loc is not inside any objects
        '''
        # if any object has the point in its interior, it is occupied
        
        for l in self.landmarks:
            if l.polygon.intersects(loc):
                return True
        # none of he objects had loc inside it
        return False

    def isOnMap(self, loc):
        '''
        loc: Point object to test if is allowed position on the map 
        returns True if loc is not inside any objects
        '''
        # if any object has the point in its interior, it is occupied
        return not(loc.x < 0 or loc.y < 0 or loc.x > self.width or loc.y > self.height)
            

class Robot:
    '''
    world: Map that the robot lives in
    localizer: Localizer object for storing the robots beief over poses
    '''
    def __init__(self, world, num_particles = 300):
        self.world = world
        self.localizer = Localizer(self, self.world, num_particles = num_particles)
        self.view_angle = math.radians(110)
        self.path = None

    def move(self, d_forward = 0, d_theta = 0, resample = False):
        '''
        updates the pose on the map and the localizer
        d_forward: distance to move in direction that robot already pointing
        d_theta: radians to turn the robot after applying translation
        returns nothing
        '''
        self.world.robot_pose.move_forward(d_forward, d_forward * .05)
        self.world.robot_pose.turn(d_theta, math.radians(3))
        self.localizer.update(d_forward = d_forward, d_theta = d_theta, resample = resample)

    def sense(self, resample = True):
        '''
        produces a (potentially) noisy reading of environment and
            updates localizer
        '''
        # for now: not noisy

        obs = self.world.whatCanISee()
        self.localizer.observation = obs
        self.localizer.just_sensed = True
        self.localizer.update(observation = obs, resample = resample)

    def move_and_sense(self, d_forward = 0, d_theta = 0):
        '''
        updates the pose on the map, senses from that pose and the localizer
        d_forward: distance to move in direction that robot already pointing
        d_theta: radians to turn the robot after applying translation
        returns nothing
        '''
        self.world.robot_pose.move_forward(d_forward, d_forward * .05)
        self.world.robot_pose.turn(d_theta, math.radians(1))
        obs = self.world.whatCanISee()
        self.localizer.update(d_forward = d_forward, d_theta = d_theta, observation = obs)

class Localizer:
    '''
    poses: list of poses
    map: the Map object of the robot
    num_particles: the size of the list of poses to maintain
    robot: the robot
    sensing_error_lambda: mean of prob distribution of sensing error (Poisson)
                          TODO: choose this value!
    '''

    def __init__(self, my_robot, my_map, confusion_matrix = None, num_particles = 150, sensing_error_lambda = 0.0001):
        self.robot = my_robot
        self.map = my_map
        self.num_particles = num_particles
        self.observation = None
        self.just_sensed = False
        if confusion_matrix == None:
            confusion_matrix = {}
            confusion_matrix['house'] = {'house': .95, 'tree' : .04, 'lake' : .01}
            confusion_matrix['tree'] = {'house': .08, 'tree' : .9, 'lake' : .02}
            confusion_matrix['lake'] = {'house': .01, 'tree' : .04, 'lake' : .95}
        self.confusion_matrix = confusion_matrix
        self.sensing_error_lambda = sensing_error_lambda
        self.num_motion_samples = 1
        self.initialization_multiplier = 2
        self.sigma_forward = 3.
        self.sigma_angle = math.radians(5)


        self.poses = self.initialize_poses()

    def initialize_poses(self):
        '''
        randomly sample over the map until self.num_poses exist. assumes uniform distribution
        returns list of poses
        '''
        # should we only sample from non-occupied regions?
        poses = []
        weight = 1.0 / self.num_particles / self.initialization_multiplier
        while len(poses) < self.initialization_multiplier * self.num_particles:
            # uncomment for not cheating initialization
            new_pose = Pose(
                random.random() * self.map.width,
                random.random() * self.map.height,
                random.random() * 2 * math.pi,
                weight
                )
            # uncomment for tight initialization
            # new_pose = Pose(
            #     (random.random() * 100 - 50) + self.map.robot_pose.x,
            #     (random.random() * 100 - 50) + self.map.robot_pose.y,
            #     random.random() * 2 * math.pi,
            #     weight
            #     )
            if not self.map.isOccupied(Point(new_pose.x, new_pose.y)):
                poses += [new_pose]
        return poses

    def resample(self):
        '''
        selects num_particles poses using sampling from CDF
        Function is O(n log n) 
        '''
        new_poses = []
        already_sampled = set([])
        count = 0
        self.poses.sort(key = lambda pose: pose.log_prob)
        cdf = numpy.cumsum([pose.prob for pose in self.poses])
        N = 2
        R = self.num_particles / N
        rands = numpy.random.random(R)
        indices = [self.inv_cdf(r + R*i, cdf) for r in rands for i in range(N)]
        if len(self.poses) > 0:
            self.poses = [self.poses[i] for i in indices]
        else:
            self.poses = self.initialize_poses()

    def inv_cdf(self, randn, cdf, low = 0, high = None):
        if high == None:
            high = len(cdf)
        if high-low <= 1:
            return low
        middle = int((high + low)/2.)
        if cdf[middle] > randn:
            return self.inv_cdf(randn, cdf, low, middle)
        else:
            return self.inv_cdf(randn, cdf, middle, high)

    def normalize_poses(self):
        '''
        reweights posese so Sum probs = 1
        '''
        if len(self.poses) > 0:
            log_probs = [pose.log_prob for pose in self.poses]
            max_lp = max(log_probs)
            scores = [math.exp(lp - max_lp) for lp in log_probs]
        
            sum_scores = sum(scores)
            for i in range(len(self.poses)):
                a = self.poses[i].prob
                self.poses[i].prob = scores[i] / sum_scores
                try:
                    self.poses[i].log_prob = math.log(scores[i]) - math.log(sum_scores)
                except:
                    print(a)

    def motion_update(self, pose, d_forward, d_theta, in_place = True):
        '''
        make a noisy update of the pose and incorportate transition
            probability p(x_new | x_old, d_forward, d_theta)
        returns nothing
        '''
        self.just_sensed = False
        if in_place:
            pose.move_forward(d_forward, self.sigma_forward)
            pose.turn(d_theta, self.sigma_angle)
        else:
            new_pose = pose.move_forward(d_forward, self.sigma_forward, in_place = False)
            new_pose.turn(d_theta, self.sigma_angle)
            return new_pose


    def sensor_update(self, pose, actual_observation):
        '''
        make an update of the probability p(observation | pose)
        '''
        expected_observation = self.map.whatCanISee(pose)
        log_error_prob = self.log_prob_observation(actual_observation, expected_observation)
        pose.log_prob += log_error_prob
        pose.prob = math.exp(pose.log_prob)

    def old_log_prob_observation(self, obs, expected_obs):
        '''
        compute log p(actual | expected)
        actual: output of whatCanISee for real robot pose
        expected: output of whatCanISee for particle
        returns log p(actual | expected)
        '''

        #TODO: actually implement, not just this BS 
        return  (len(expected_obs))* math.log(.8) + abs(len(expected_obs) - len(obs)) * math.log(.2)


    def log_prob_observation(self, actual_obs, expected_obs):
        '''
        Observation Model for a Random Number of Detections
        Described in the paper.
        '''
        log_permutation_sums = []
        for permutation in self.generate_pi(actual_obs):
            log_numerator = 0
            log_denominator = 0
            # mtraub: I added the min because sometimes len(perm) < len(exp_obs), throwing error
            # TODO: evaluate iif this is the right fix
            for i in range(min(len(expected_obs), len(permutation))):
                if permutation[i] != 0:
                    log_numerator += self.log_prob_detection(permutation[i]) + self.log_prob_single_obj_observation(permutation[i],expected_obs[i]) 
                    log_denominator += math.log((1 - math.exp(self.log_prob_detection(permutation[i])))) + math.log(self.sensing_error_lambda) + self.log_prob_noisy_detection(permutation[i])
            log_permutation_sums += [log_numerator - log_denominator]
        log_permutation_sums = numpy.array(log_permutation_sums)
        max_lps = max(log_permutation_sums)
        permutation_sum = numpy.sum(numpy.exp(log_permutation_sums - max_lps)) * math.exp(max_lps)
        return self.log_prob_observation_all_noise(actual_obs) + \
               self.log_prob_observation_all_missed_detections(actual_obs) + \
               math.log(permutation_sum)
    
    def log_prob_noisy_detection(self, obs_obj):
        '''
        Assumed for now to be uniform
        kappa(z) = 1/(|C||S||B|)
        this is a uniform distribution over
        - the number of classes
        - the potential scores of a classification ()
        - the potential bearings of an object in FoV
        essentially representing even likelihood of a detection
        of any class of any score at any location in the FoV
        '''
        return -math.log(len(self.map.classes)) - math.log(1.0) - math.log(self.robot.view_angle)

    def log_prob_observation_all_noise(self, obs):
        '''
        A poisson distribution over the set of detections, according to set
        integrals this will have a PDF 1 like a proper distribution

        Assumes that all noise is representable as a poisson distribution
        '''
        clutter_log_prob = sum([math.log(self.sensing_error_lambda) + self.log_prob_noisy_detection(obs_obj) for obs_obj in obs])
        return self.sensing_error_lambda + clutter_log_prob

    def log_prob_observation_all_missed_detections(self, obs):
        '''
        The unfortunate case that for a given observation, that all the observed
        objects are unsuccessfully detected

        essentially, the probability that all objects in obs are not seen
        '''
        return sum([math.log(1 - math.exp(self.log_prob_detection(obj))) for obj in obs])

    def generate_pi(self, obs):
        '''
        pi is a function that generates all of the potential classifications
        for the set of objects in a given observation

        what this is is for a given observation, we return an iterator over all
        permutations of the n objects in that observation and potentially noise
        replacing any of them mapped to the n observed classes, representing all
        the ways we could see those n classes in the frame
        '''
        return itertools.permutations(obs + [0]*len(obs), len(obs))


    def log_prob_single_obj_observation(self, obs_obj, expected_obs_obj, score=1):
        '''
        Observation Model for a Single Object Detection.
        TODO: WHAT IS THIS SCORE EVEN
        '''
        log_confusion_prob = self.log_prob_confusion(obs_obj.class_type, expected_obs_obj.class_type)
        log_bearing_prob = self.log_prob_bearing(obs_obj, expected_obs_obj)
        log_score_prob = self.log_prob_score(score)
        return log_confusion_prob + log_bearing_prob + log_score_prob


    def log_prob_confusion(self, obs_obj_class, expected_obj_class):
        '''
        The probability of confusing obj_class for expected_obj_class.
        We can decide on the actual distributition afterwards.
        '''
        return math.log(self.confusion_matrix[obs_obj_class][expected_obj_class])

    def log_prob_bearing(self, obs_obj, expected_obj):
        '''
        pdf p_beta(.| y, x) as that of a Gaussian distribution with
        mean beta(x, y) and covariance beta
        '''
        #TODO: Write this
        sigma = math.radians(8)
        return -math.log(2 * math.pi * sigma) - ((obs_obj.theta - expected_obj.theta) ** 2) / (sigma ** 2)

    def log_prob_score(self, obs_score, obj_class = None, expected_obj_class = None):
        '''
        This should be the "score" of a given detection of an object
        from obeservation, an arbitrary value assigned to the detection
        for which there exists some distribution over all possible values
        and the likelihood that an object with a given score has been
        detected correctly

        Since we are "simulating" vision, we will also simulate this score
        by just giving a score 1 to any object for the class it was detected as
        and 0 for any class it was not detected as, essentially representing
        perfect classification
        '''
        return math.log(1.0)

    def log_prob_detection(self, obj):
        '''
        Probability of detection an obj given pose. Dependent on distance.
        Assumes that obj is in field of view from pose.
        '''
        # TODO: define p_{d, 0} and std_d
        # p_{d,0} and std_d should be based on class according to literature
        # p_d0 = 1
        # std_d = 200
        # return math.log(p_d0) - (obj.dist ** 2) / (std_d ** 2)
        return math.log(.5)

    def most_likely_pose(self):
        '''
        return the highest prob pose
        '''
        ml_pose = pose[0]
        for pose in self.poses:
            if pose.prob > ml_pose.prob:
                ml_pose = pose
        return ml_pose

    def expected_location(self):
        '''
        return the average of positions as Point
        '''
        x = 0
        y = 0
        self.normalize_poses()
        for pose in self.poses:
            x += pose.x * pose.prob
            y += pose.y * pose.prob
        return Point(x, y)
