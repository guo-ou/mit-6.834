#!/usr/bin/env python

import rospy
import actionlib

from activity_template.msg import ActivityExampleAction, ActivityExampleGoal, ActivityExampleResult, ActivityExampleFeedback

class Activity(object):
    def __init__(self, activity_name='activity_name'):
        self.action_server = actionlib.SimpleActionServer(activity_name, ActivityExampleAction, self.execute_action, False)
        self.action_server.start()

    def execute_action(self, goal):
        # YOUR CODE HERE! Replace this method.
        rospy.loginfo("Action called with input: {}".format(goal.input))


        rospy.sleep(rospy.Duration(1.0))



        # This is how you can send feedback to the client
        feedback = ActivityExampleFeedback()
        feedback.percent_complete = 0.5
        self.action_server.publish_feedback(feedback)



        rospy.sleep(rospy.Duration(1.0))


        # Before you exit, be sure to succeed or fail.
        result = ActivityExampleResult()
        result.output = "I'm done!" # Put your output here
        self.action_server.set_succeeded(result)



if __name__ == '__main__':
    rospy.init_node('activity_example_server')
    activity = Activity(activity_name='activity_name')
    rospy.loginfo('Activity example running')
    rospy.spin()
