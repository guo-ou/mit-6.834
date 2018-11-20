# 16.412/6.834 Cognitive Robotics Advanced lecture: Semantic Localization
import math
import numpy
import random
import scipy.stats as stats

'''
This file contains geometric structures for computational geometry calculations
'''

class Pose:
    '''
    x: x coordinate of position
    y: y coordinate of position
    theta: bearing, def CCW from +X in radians
    prob: probability of being in that point
    log_prob: will probably be needed to avoid underflow
    '''
    def __init__(self, x, y, theta = 0, prob = None, log_prob = None):
        self.x = x
        self.y = y
        self.theta = theta
        if prob == None and log_prob == None: #default prob values
            self.prob = 1
            self.log_prob = 0
        elif prob == None: # if only log_prob given
            if log_prob <= 0:
                self.log_prob = log_prob
                self.prob = math.exp(log_prob)
            else:
                raise ValueError( str(log_prob) + " is an invalid log-probability.")
            
        elif log_prob == None: #if only prob given
            if prob > 0 and prob <= 1:
                self.log_prob = math.log(prob)
                self.prob = prob
            elif prob == 0: 
                self.log_prob = -numpy.inf
                self.prob = 0
            else:
                raise ValueError( str(prob) + " is an invalid probability.")
        else: # if both given make sure they match up
            if math.exp(log_prob) == prob:
                self.prob = prob
                self.log_prob = log_prob
            else:
                raise ValueError( "prob (" + str(prob) + ") and log_prob (" + str(log_prob) + ") do not correspond to the same probability.")

    def sample(self):
        '''
        randomly sample pose at probability of self.prob
        returns: True or False
        '''
        return random.random() <= self.prob


    def move_forward(self, dx, sigma_noise = 0, in_place = True):
        '''
        moves the robot pose forward dx with sigma_noise and changes
            log_prob accordingly
        returns new Pose if update = True
        '''
        if sigma_noise == 0:
            noise = 0
            log_prob_movement = 0
        else:
            noise = random.gauss(0,sigma_noise)
            log_prob_movement = stats.norm.logpdf(noise, 0, sigma_noise)

        if in_place:
            self.x += (dx + noise) * math.cos(self.theta) 
            self.y += (dx + noise) * math.sin(self.theta)
            self.add_log_prob(log_prob_movement)
        else:
            x = self.x + (dx + noise) * math.cos(self.theta) 
            y = self.y + (dx + noise) * math.sin(self.theta)
            log_prob = self.log_prob + log_prob_movement
            return Pose(x = x, y = y, theta = self.theta, log_prob = log_prob)


    def turn(self, d_theta, sigma_noise = 0, in_place = True):
        '''
        rotates the robot pose d_theta with sigma_noise and changes
            log_prob accordingly
        returns new Pose if update = True
        '''

        if sigma_noise == 0:
            noise = 0
            log_prob_movement = 0
        else:
            noise = random.gauss(0, sigma_noise)
            log_prob_movement = stats.norm.logpdf(noise, 0, sigma_noise)

        if in_place:
            self.theta = (self.theta + d_theta + noise) % math.radians(360)
            self.add_log_prob(log_prob_movement)
        else:
            theta = (self.theta + d_theta + noise) % math.radians(360)
            log_prob = self.log_prob + log_prob_movement
            return Pose(x = self.x, y = self.y, theta = theta, log_prob = log_prob)

    def multiply_prob(self, times_prob):
        '''
        update prob to be prob * times_prob, and log_prob accordingly
        '''
        self.add_log_prob(math.log(times_prob))

    def add_log_prob(self, plus_log_prob):
        """
        update log_prob to be log_prob + plus_log_prob, and prob accordingly
        """
        try:
            self.log_prob += plus_log_prob
            self.prob = math.exp(self.log_prob)
        except:
            print("*", self.prob, self.log_prob, plus_log_prob)
            self.prob = 0
            self.log_prob = -numpy.inf


    def bearing_to(self, x, y):
        dx = x - self.x
        dy = y - self.y
        if dx == 0:
        	angle = math.radians(90) * dy / abs(dy)
        else:
        	angle =  math.atan(dy / dx)
        	if dx < 0:
        		angle += math.radians(180)
        return angle - self.theta 

    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)


