import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
import threading

"""
Get pointclouds from ros system, convert them to proper format, then classify using
voxelnet, finaly output bounding boxes.

"""
POINTCLOUD_FILE = "pc_input.p"
BBOX_FILE       = "bbox_output.p"

pc2_msg = None
lock    = None

def pc_callback(msg):
    global pc2_msg
    global lock

    lock.acquire()
    pc2_msg = msg
    lock.release()


def car_detector():
    rospy.init_node('car_detector',anonymous=False)
    sub = rospy.Subscriber('/velodyne_points_right',PointCloud2,pc_callback)

    while rospy
