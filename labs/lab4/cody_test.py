from math import pi
import numpy as np
import rospy
import tf
import sys

from geometry_msgs.msg import TwistStamped

from core.interfaces import ArmController

import sys
import rospy
import numpy as np
from collections import namedtuple
from time import perf_counter

from core.interfaces import ArmController
from lib.potentialFieldPlanner import PotentialFieldPlanner 

from copy import deepcopy

#########################
##  RViz Communication ##
#########################

rospy.init_node("visualizer")

twist_pub = rospy.Publisher('/vis/twist', TwistStamped, queue_size=10)
joint_pub = rospy.Publisher('/vis/jointvel', TwistStamped, queue_size=10)
listener = tf.TransformListener()

def qv_mult(q, v):
    v = [v[0],v[1],v[2],0]
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q, v),
        tf.transformations.quaternion_conjugate(q)
    )[:3]

# Publishes the linear and angular velocity of a frame on the corresponding topic
def show_twist(pub, velocity, frame):
    (trans,rot) = listener.lookupTransform('world', frame, rospy.Time(0))
    quat = tf.transformations.quaternion_conjugate(rot)
    velocity = qv_mult(quat,velocity[0:3])
    msg = TwistStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame
    msg.twist.linear.x  = velocity[0]
    msg.twist.linear.y  = velocity[1]
    msg.twist.linear.z  = velocity[2]
    msg.twist.angular.x = 0
    msg.twist.angular.y = 0
    msg.twist.angular.z = 0
    pub.publish(msg)

# Publishes the velocity of a given joint on the corresponding topic
def show_joint_velocity(i, lin_v):
    # joints are always along z axis
    show_twist(joint_pub, lin_v, 'joint' + str(min(i, 6)))



start_config = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
goal_config = np.array([0,0,0,-pi/2,0,pi,pi/4])

if __name__ == "__main__":

    arm = ArmController()
    print("Moving to Start Position")
    arm.safe_move_to_position(start_config)

    print("Starting to plan")


    MyStruct = namedtuple("map", "obstacles")
    map_struct = MyStruct(obstacles = np.array([]))

    planner = PotentialFieldPlanner()
    path, force_path = planner.plan(deepcopy(map_struct), deepcopy(start_config), deepcopy(goal_config))

    input("Press Enter to Send Path to Arm")

    gap = 50
    for i in range(0, path.shape[0], gap):
        arm.safe_move_to_position(path[i, :])
        forces = force_path[i]
        for index in range(3, forces.shape[1]):
            show_joint_velocity(index, forces[:, index])
            print(forces[:, index])
            input("press enter to continue")

    print("Trajectory Complete!")