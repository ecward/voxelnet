#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
import threading
import cPickle as pickle
import numpy as np
import mmap
import contextlib

pc2_msg      = None
pc2_msg_idx  = None
pc2_msg_lock = None

def pc2_callback(msg):
    global pc2_msg
    global pc2_msg_idx
    global pc2_msg_lock

    print "got msg!"
    pc2_msg_lock.acquire()
    if pc2_msg_idx is None:
        pc2_msg_idx = 0
    else:
        pc2_msg_idx += 1
    pc2_msg = msg
    pc2_msg_lock.release()


def get_R(roll,pitch,yaw):
    cos_alph = np.cos(yaw)
    sin_alph = np.sin(yaw)
    cos_beta = np.cos(pitch)
    sin_beta = np.sin(pitch)
    cos_gamma = np.cos(roll)
    sin_gamma = np.sin(roll)
    
    return np.array(
        [[cos_alph*cos_beta, 
         cos_alph*sin_beta*sin_gamma-sin_alph*cos_gamma,
         cos_alph*sin_beta*cos_gamma+sin_alph*sin_gamma],
        [sin_alph*cos_beta, 
         sin_alph*sin_beta*sin_gamma+cos_alph*cos_gamma,
        sin_alph*sin_beta*cos_gamma-cos_alph-cos_alph*sin_gamma],
        [-sin_beta,cos_beta*sin_gamma,cos_beta*cos_gamma]])

#10 MB
FILESIZE_PC = 10000000
#10 kb
FILESIZE_BBOX = 10000
POINTCLOUD_FILE = "pc_input.p"
BBOX_FILE       = "bbox_output.p"

if __name__=="__main__":
    rospy.init_node("car_detector_ros")

    #comms
    pc2_msg_lock = threading.Lock()

    #transform pointcloud to baselink, TODO, don't hard-code
    
    #From tf-echo
    translation = np.array([0.030, 0.426, -0.158])
    rot = get_R(0,-0.211,-1.571)


    #Todo listen to fused clouds
    sub = rospy.Subscriber('/velodyne_points_right',PointCloud2,pc2_callback)


    with open(POINTCLOUD_FILE,"r+b") as pc_f:
        with open(BBOX_FILE,"rb") as bbox_f:

            r = rospy.Rate(10) # 10hz

            #let's be careful with mmap files
            buf_pc = mmap.mmap(pc_f.fileno(),0,mmap.MAP_SHARED,mmap.PROT_WRITE)

            #main loop
            while not rospy.is_shutdown():

                pc2_msg_lock.acquire()
                points_idx = pc2_msg_idx
                points = []
                if pc2_msg is not None:
                    for point in pcl2.read_points(pc2_msg,skip_nans=True):
                        points.append(point)
                pc2_msg_lock.release()

                if len(points) == 0:
                    print("No points..")
                    r.sleep()
                    continue
                else:
                    print("Got points at idx =",points_idx)

                points_np = np.array(points)
                points_sbl = np.dot(points_np[:,:3],np.linalg.inv(rot).T)-translation



                #Fake intensity = 1 and store as flat array
                points_xyzi = np.ones([np.size(points_sbl,0),4],dtype=np.float32)
                points_xyzi[:,:3] = points_sbl

                #In kitty the sensor is about 0.5 higher than this, so let's remove
                #-0.5 from z
                points_xyzi[:,2] -= 0.5

                print "Dumping pointcloud with size",np.shape(points_xyzi),\
                    " idx = ",points_idx,\
                    " to mmaped file..."
                buf_pc.seek(0)
                pickle.dump((points_idx,points_xyzi.flatten()),buf_pc)
                buf_pc.flush()
                print "done!"

                #with contextlib.closing(mmap.mmap(pc_f.fileno(),0,access=mmap.ACCESS_WRITE)) as m:
                 #   m.seek(0) #rewind
                 #   pickle.dump((points_idx,points_xyzi.flatten()),m)
                 #   m.flush()
                 #   print "done!"
                #Check if we have new results.... TODO

                r.sleep()
