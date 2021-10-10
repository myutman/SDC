#! /usr/bin/python3

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import time
import numpy as np


class TracingTurtle:

    def __init__(self):
        self.publisher = rospy.Publisher('/micheliangelo/cmd_vel', Twist, queue_size=10)
        self.subscriber_me = rospy.Subscriber('/micheliangelo/pose', Pose, self.callback_me)
        self.subscriber_other = rospy.Subscriber('/turtle1/pose', Pose, self.callback_other)
        self.rate = rospy.Rate(0.01)

    def callback_me(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.theta = msg.theta

    def callback_other(self, msg):
        x = msg.x
        y = msg.y
        dx = x - self.x
        dy = y - self.y
        theta1 = np.arctan2(dx, dy)
        dtheta = theta1 - self.theta
        
        msg1 = Twist()
        msg1.linear.x = dx
        msg1.linear.y = dy

        msg2 = Twist()
        msg2.angular.z = dtheta

        self.publisher.publish(msg2)
        #self.rate.sleep()
        self.publisher.publish(msg1)
        #self.rate.sleep()



rospy.init_node('turtle_tracing')
tracing_turtle = TracingTurtle()
rospy.spin()

