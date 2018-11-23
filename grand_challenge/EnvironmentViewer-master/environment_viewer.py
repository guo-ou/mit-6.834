#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import scipy.misc
import argparse
import yaml

class EnvironmentViewer():
    """ Displays an Enterprise environment represented by a YAML file """
    def __init__(self):
        # Datastructures
        self.obstacles = {}
        self.features = {}
        # For overlaying a ROS-format map
        self.overlay_meta = None
        self.overlay_image = None

    def load_from_file(self, filename):
        # Read & parse the YAML file
        with open(filename, 'r') as f:
            d = yaml.load(f)

        self.obstacles = d['environment']['obstacles'] if 'obstacles' in d['environment'] else {}
        self.features = d['environment']['features'] if 'features' in d['environment'] else {}

    def load_ros_map(self, filename_yaml, filename_image):
        with open(filename_yaml, 'r') as f:
            d = yaml.load(f)
        self.overlay_meta = d
        self.overlay_image = filename_image

    def _display_obstacle(self, ax, obs):
        self._draw_shape(ax, obs, color=(1.0, 0.0, 0.0))

    def _display_feature(self, ax, feature):
        self._draw_shape(ax, feature, color=(0.0, 0.0, 1.0))


    def _draw_shape(self, ax, desc, color='g'):
        if desc['shape'] == 'polygon':
            vertices = np.array(desc['corners'])
            polygon = Polygon(vertices, closed=True, facecolor=color, linewidth=2.0, edgecolor='k', alpha=0.5)
            ax.add_patch(polygon)

        elif desc['shape'] == 'rectangle':
            center = (desc['center'][0] - desc['width'] / 2.0, desc['center'][1] - desc['length'] / 2.0)
            rotation = desc['rotation'] if 'rotation' in desc else 0.0
            rectangle = Rectangle(center, width=desc['width'], height=desc['length'], angle=rotation, facecolor=color, linewidth=2.0, edgecolor='k', alpha=0.5)
            ax.add_patch(rectangle)

    def _draw_overlay(self):
        img = scipy.misc.imread(self.overlay_image)
        (w_pix, h_pix) = np.shape(img)
        x_left = self.overlay_meta['origin'][0]
        y_bottom = self.overlay_meta['origin'][1]
        x_right = x_left + w_pix * self.overlay_meta['resolution']
        y_top = y_bottom + h_pix * self.overlay_meta['resolution']
        # Compute extents
        plt.imshow(img, zorder=0, extent=[x_left, x_right, y_bottom, y_top], cmap='gray')

    def display(self):
        fig, ax = plt.subplots()
        # Display overlay, if loaded
        if self.overlay_meta:
            self._draw_overlay()
        # Display obstacles
        for (obs_name, obs) in self.obstacles.iteritems():
            self._display_obstacle(ax, obs)
        # Display features
        for (feature_name, feature) in self.features.iteritems():
            self._display_feature(ax, feature)

        ax.autoscale_view()
        fig.canvas.mpl_connect('button_press_event', self._onclick)
        plt.axis('equal')
        plt.title('Map Viewer')
        plt.show()

    def _onclick(self, event):
        print('        - [{}, {}]'.format(event.xdata, event.ydata))



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Displays an environment stored in a file')
    parser.add_argument('environment', type=str, help='The file to display (ends in .yaml)')
    parser.add_argument('--ros-map', type=str, help='An optional ROS map file (ends in .pgm)', default=None)
    parser.add_argument('--ros-yaml', type=str, help='An optional ROS yaml map file (ends in .yaml)', default=None)
    args, unknown = parser.parse_known_args()
    # Go!
    ev = EnvironmentViewer()
    ev.load_from_file(args.environment)
    if args.ros_map and args.ros_yaml:
        ev.load_ros_map(args.ros_yaml, args.ros_map)
    ev.display()
