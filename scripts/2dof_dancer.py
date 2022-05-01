#!/usr/bin/env python3

# 
# 2dof_dancer.py
# 
# Outputs a dancing pattern to the 2-dof arm. 
#
# 18 April 2022
# CS 230 Final Project
#

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np

rospy.init_node("dancer", anonymous=True)
dance_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
string_pub = rospy.Publisher("string_channel", String, queue_size=10)

rate = rospy.Rate(10)
joint1 = 0
joint2 = 0
while not rospy.is_shutdown():
    t = rospy.get_rostime()
    joint1 = (.5 * np.sin(3 * (t.now().secs + 1.E-9 * t.now().nsecs)) +
            .2 * (t.now().secs + 1.E-9 * t.now().nsecs))
    joint2 = .000000003* (np.sin(1 * (t.now().secs + 1.E-9 * t.now().nsecs)) * 
            (t.now().secs + 1.E-9 * t.now().nsecs))
    
    js = JointState()
    js.name = ["joint_base_link__link1", "joint_link1__link2"]
    js.position = [joint1, joint2]
    js.header.stamp.secs = t.now().secs
    js.header.stamp.nsecs = t.now().nsecs
    
    s = String("abc")
    
    dance_pub.publish(js)
    string_pub.publish(s)
    rate.sleep()