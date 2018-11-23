Environment & Environment Viewer
============================

We'll be representing the map / environment in the grand challenge in two ways, depending on which is more convenient for your group.


ROS Maps
----------
The first way is via a grayscale bitmap image (a .pgm file), which intuitively represents an occupancy grid. White cells are free, black pixels are obstacles, and grey cells are  unknown. This is viewable in the `map.pgm` file (which our implementation built using SLAM via the ROS `gmapping` package).

There is also an associated `map.yaml` file, which has some additional information about the image, such as how "big" each pixel is (in meters per pixel, the pixel location location of where the physical (0, 0) world is represented within that image, etc.).

We expect this map format to be particularly useful for the incremental path planning team.


Environment YAML
---------------------
In this representation, we represent the world not as an image but rather more semantically. Namely, we represent obstacles / walls in the world as polygons. We also represent special regions of interest called `features` as polygons as well, such as the home location, the locations of the three challenge stations, etc.

All of this information is specified in a an `environment.yaml` file (which is distinct from the `map.yaml` above). This file is human readable, so you should be able to easily read it and see what's going on.


Map Viewer
------------
To make your life easier, we've providing a map visualizer. This will help you see the map of the grand challenge, and also allow you to better debug if you need to edit / make any changes to the map. This viewer is callable via the `environment_viewer.py` script.

To run it and visualize an `environment.yaml` file, please run:
```
./environment_viewer.py environment.yaml
```
That will bring up a GUI in which you can inspect the map. You can pan and zoom. Also, if you click somewhere in the map, the coordinates of where you clicked will be printed out to the terminal (this is useful when adding new obstacles or features).

The above command only visualizes the `environment.yaml` file. You can additionally overlay the ROS `map.pgm` file underneath to make sure that they match up. This can be accomplished with some additional command line arguments:

```
./environment_viewer.py environment.yaml --ros-map map.pgm --ros-yaml map.yaml
```

Good luck! As usual, let the course staff know if you have any questions.
