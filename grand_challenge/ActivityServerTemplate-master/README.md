ROS Activity Template
=======================

This template provides an example of a ROS actionlib server, which will be used to integrate some group's grand challenge projects into the MERS planning & execution framework (called Enterprise) that will be controlling the robot.

The `activity_template` folder is a ROS package. First, make sure ROS is installed on your system, and then create a workspace. Instructions on installing ROS can be found here (please use the `indigo` version on an Ubuntu 14.04-esque machine such as the course VM): http://wiki.ros.org/indigo/Installation/Ubuntu

More ROS tutorials, including those for setting up your ROS workspace, can be found here:
http://wiki.ros.org/ROS/Tutorials

Then, copy this folder into your ROS workspace. Assuming you have a setup similar to that described by the online ROS tutorials, this can be accomplished by copying the folder into `~/catkin_ws/src/`.

Once the folder is placed there, make sure your workspace is built:
```
cd ~/catkin_ws/
catkin_make
```
(you shouldn't have any errors).

Once that's done, you're ready to test this template! In a terminal window, please run:
```
roslaunch activity_template activity_server_example.launch
```

This will start a template activity server. Think of this as a program that sits around and waits, until a request is made. Then, it'll start doing that request, publish any feedback or updates as appropriate, and tell you when the task finishes or fails (Note that we're using ROS [actionlib](http://wiki.ros.org/actionlib) to implement this server). Your job will be to replace the appropriate method in that server to call your code, and also to update the `ActivityExample.action` file as appropriate to represent any input / output you need.

The Enterprise planning & execution framework will call your action server at the appropriate time, to signal that your code should start running to perform a request (e.g., play a game, classify a picture, do a reachability challenge, etc.).

We've also provided you with a sample script to call the activity server to help you debug. Once the above server is running, you can try calling it by running this in a new terminal window:

```
rosrun activity_template activity_client.py
```

When run, this will start a ROS node that calls the server once, and then exits.

Good luck! As always, let the course staff know if you have any questions.
