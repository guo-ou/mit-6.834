#! /usr/bin/env python

import rospy
import actionlib

from activity_template.msg import ActivityExampleAction, ActivityExampleGoal, ActivityExampleResult

def activity_client(activity_name='activity_name'):
    # Creates the SimpleActionClient, passing the type of the action
    # (ActivityExampleAction) to the constructor.
    client = actionlib.SimpleActionClient(activity_name, ActivityExampleAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    # Creates a goal to send to the action server.
    goal = ActivityExampleGoal()
    goal.input = "Here's a problem for the server to solve."

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # An ActivityExampleResult

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('activity_example_client')
        result = activity_client()
        print "Output: {}".format(result.output)
    except rospy.ROSInterruptException:
        print "Program interrupted before completion!"
