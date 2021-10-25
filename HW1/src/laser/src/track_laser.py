#! /usr/bin/python3

import numpy as np

import rospy
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan


rospy.init_node('viz_map')

class LaserCallback:
    
    def __init__(self):
        self.marker_publisher = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.map_publisher = rospy.Publisher('/map_topic', OccupancyGrid, queue_size=10)
        self.rate = rospy.Rate(20)

    def __call__(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        alphas = msg.angle_min + msg.angle_increment * np.arange(len(msg.ranges))

        step = 0.05
        small_diffs = np.abs(ranges[2:] - ranges[:-2]) < step
        is_ok = np.concatenate([[True, True], small_diffs]) | np.concatenate([small_diffs, [True, True]])

        xs = ranges * np.cos(alphas)
        ys = ranges * np.sin(alphas)

        xs = xs[is_ok]
        ys = ys[is_ok]

        
        marker = Marker()

        marker.id = 0
        marker.action = 0
        marker.header.frame_id = "base_laser_link"

        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0

        marker.color.r = 0.5
        marker.color.g = 0.1
        marker.color.b = 1.0
        marker.color.a = 0.7

        marker.type = marker.POINTS

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
 
        marker.points = [Point(x, y, 0.0) for x, y in zip(xs, ys)]

        self.rate.sleep()

        self.marker_publisher.publish(marker)
        

        grid = OccupancyGrid()

        grid.header.frame_id = 'base_laser_link'

        grid.info.resolution = step
        radius = 9
        grid.info.width = 2 * int(radius // step) + 3
        grid.info.height = 2 * int(radius // step) + 3

        grid_data = np.zeros((grid.info.width, grid.info.height), dtype=int)

        for x, y in zip(xs, ys):
            if abs(x) < radius and abs(y) < radius:
                i = int((x + radius) // step)
                j = int((y + radius) // step)
                grid_data[i][j] = min(grid_data[i][j] + 10, 100)
        
        range_x = np.arange(grid.info.width) * step - radius
        range_y = np.arange(grid.info.height) * step - radius

        grid.info.origin.position.x = - radius
        grid.info.origin.position.y = - radius
        grid.info.origin.position.z = 0.0

        self.rate.sleep()
        grid.header.seq = i
        grid.data = list(grid_data.transpose(1, 0).reshape(-1))
        self.map_publisher.publish(grid)


subscriber = rospy.Subscriber('/base_scan', LaserScan, LaserCallback())

rospy.spin()


