import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patches

landmark_colors = {'house': 'k', 'tree': 'g', 'lake' : 'b'}

def simulate(command, world, localizer):  
    fig, ax = plt.subplots()
    ax.set_xlim([0,world.width])
    ax.set_ylim([0,world.height])
    draw_landmarks(ax, world.landmarks)
    Q_particles = draw_particles(localizer.poses)
    Q_robot = draw_robot(world.robot_pose)
    detection_pts = []
    ani = anim.FuncAnimation(fig, update, fargs = (world, localizer, command, Q_particles, Q_robot, detection_pts), interval=1000)
    plt.show()

def draw_landmarks(ax, landmarks):
    
    for landmark in landmarks:
        x_pts = [pt[0] for pt in landmark.polygon.exterior.coords]
        y_pts = [pt[1] for pt in landmark.polygon.exterior.coords]
        fill = patches.Polygon(np.array([x_pts, y_pts]).transpose(), color = landmark_colors[landmark.get_class()], alpha = .3)
        ax.add_patch(fill)
        plt.plot(x_pts, y_pts, color=landmark_colors[landmark.get_class()], alpha=0.7,
            linewidth=2, solid_capstyle='round', zorder=2)
    

def draw_path(path):
    for i in range(len(path)):
        x_from, y_from = path[i]
        x_to, y_to = path[(i + 1) % len(path)]
        plt.arrow(x_from, y_from, x_to - x_from, y_to - y_from, color = "purple", head_width = 5)

def draw_particles(poses):
    x = [pose.x for pose in poses]
    y = [pose.y for pose in poses]
    u = [math.cos(pose.theta) for pose in poses]
    v = [math.sin(pose.theta) for pose in poses]
    return plt.quiver(x, y, u, v, color='r') 

def draw_robot(pose):
    return plt.quiver([pose.x], [pose.y], [math.cos(pose.theta)], 
        [math.sin(pose.theta)], color='b')

def update_particles(Q, poses):
    x = [pose.x for pose in poses]
    y = [pose.y for pose in poses]
    max_prob = max([pose.prob for pose in poses]) if poses else 0
    min_prob = min([pose.prob for pose in poses]) if poses else 0
    range_prob = max_prob - min_prob
    #print [math.pow(.1 + abs(pose.prob / range_prob), 1./3) for pose in poses]
    u = [math.cos(pose.theta) for pose in poses]
    v = [math.sin(pose.theta) for pose in poses]

    alphas = np.array([pose.prob for pose in poses])
    alphas /= max(alphas)
    alphas = alphas * .8 + .2
    rgba_colors = np.zeros((len(alphas),4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas
    
    Q.set_color(rgba_colors)
    Q.set_offsets(np.column_stack([x, y]))
    Q.set_UVC(u, v)

def update_robot(Q, pose):
    Q.set_offsets(np.column_stack([pose.x, pose.y]))
    Q.set_UVC([math.cos(pose.theta)], [math.sin(pose.theta)])

def update_detections(pose, pts, new_observations):
    # remove old detections
    if new_observations != None:
        for (circle, text) in pts:
            circle.remove()
            text.remove()
        del pts[:]

        # plot new detections
        for obs in new_observations:
            detection_x = pose.x + obs.dist * math.cos(pose.theta + obs.theta)
            detection_y = pose.y + obs.dist * math.sin(pose.theta + obs.theta)
            pt = plt.plot([detection_x], [detection_y], landmark_colors[obs.class_type]+'o')[0]
            text = plt.annotate(obs.class_type, xy=(detection_x, detection_y), 
                xytext=(detection_x , detection_y + 10), ha = 'center')
            pts += [(pt, text)]
        plt.draw()

def update(num, world, localizer, command, Q_particles, Q_robot, detection_pts):
    command(num)
    update_particles(Q_particles, localizer.poses)
    update_robot(Q_robot, world.robot_pose)
    if localizer.just_sensed:
        update_detections(world.robot_pose, detection_pts, localizer.observation)