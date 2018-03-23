#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker,MarkerArray
import threading
import cPickle as pickle
import numpy as np
import mmap
import contextlib
import tf

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


#woah, was wrong but still worked without fused clouds....
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
        sin_alph*sin_beta*cos_gamma-cos_alph*sin_gamma],
        [-sin_beta,cos_beta*sin_gamma,cos_beta*cos_gamma]])

#10 MB
FILESIZE_PC = 10000000
#100 kb
FILESIZE_BBOX = 100000
POINTCLOUD_FILE = "pc_input.p"
BBOX_FILE       = "bbox_output.p"

USE_FUSED_CLOUDS = True

if __name__=="__main__":
    rospy.init_node("car_detector_ros")

    #comms
    pc2_msg_lock = threading.Lock()

    #z-offset is to get points at a similar height as kitty velo which is mounted about 1.7 meters up

    #transform pointcloud to frame, TODO, don't hard-code
    if USE_FUSED_CLOUDS:
        pc2_frame = "base_link"
        sub = rospy.Subscriber('/fused_tracking_cloud',PointCloud2,pc2_callback)
        translation = np.array([0.0, 0.0, 0.0])
        rot = get_R(0,0,0)
        z_offset = -1.5
    else:
        pc2_frame = "sensor_board_link"
        sub = rospy.Subscriber('/velodyne_points_right',PointCloud2,pc2_callback)
        translation = np.array([0.030, 0.426, -0.158])
        rot = get_R(0,-0.211,-1.571)
        z_offset = -0.5

    print "using rot = "
    print rot
    print "translation = ",translation
        

    #output bounding boxes
    marker_publisher = rospy.Publisher('voxelnet_bbox', MarkerArray,queue_size=10)
    max_num_det      = 0

    with open(POINTCLOUD_FILE,"r+b") as pc_f:
        with open(BBOX_FILE,"rb") as bbox_f:

            r = rospy.Rate(10) # 10hz

            #let's be careful with mmap files
            buf_pc   = mmap.mmap(pc_f.fileno(),0,mmap.MAP_SHARED,mmap.PROT_WRITE)
            buf_bbox = mmap.mmap(bbox_f.fileno(),0,mmap.MAP_SHARED,mmap.PROT_READ)


            #maintain a mapping from pointcloud to the time it was produced!
            pc_idx_to_rostime = {}
            
            #main loop
            while not rospy.is_shutdown():

                pc2_msg_lock.acquire()
                points_idx = pc2_msg_idx
                points = []
                if pc2_msg is not None:
                    pc_idx_to_rostime[points_idx] = pc2_msg.header.stamp
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
                points_xyzi[:,2] += z_offset

                print "Dumping pointcloud with size",np.shape(points_xyzi),\
                    " idx = ",points_idx,\
                    " to mmaped file..."
                buf_pc.seek(0)
                pickle.dump((points_idx,points_xyzi.flatten()),buf_pc)
                buf_pc.flush()
                print "done!"

                #load result
                idx_bbox = None
                result   = None
                try:
                    buf_bbox.seek(0)
                    idx_bbox,result = pickle.load(buf_bbox)
                except Exception as e:
                    print(e)
                    print("Failed to load results")

                if result is not None:
                    print "results for idx = ",idx_bbox," = ",result

                
                if result is not None:
                    max_num_det      = max(max_num_det,len(result))
                    marker_arr = MarkerArray()
                    marker_id  = 0
                    rostime = rospy.get_rostime()
                    if idx_bbox in pc_idx_to_rostime:
                        rostime = pc_idx_to_rostime[idx_bbox]


                    #add a deleteall marker first so we only show latest ones
                    bbox_marker = Marker()
                    bbox_marker.header.seq = 0 
                    bbox_marker.header.stamp = rostime
                    bbox_marker.header.frame_id = pc2_frame
                    bbox_marker.ns = 'voxelnet_bbox'
                    bbox_marker.action = Marker.DELETEALL
                    marker_arr.markers.append(bbox_marker)

                    for bbox in result:
                        bbox_marker = Marker()
                        bbox_marker.header.seq = 0 
                        bbox_marker.header.stamp = rostime
                        bbox_marker.header.frame_id = pc2_frame
                        bbox_marker.ns = 'voxelnet_bbox'
                        bbox_marker.id = marker_id
                        marker_id += 1                        
                        bbox_marker.type = Marker.CUBE
                        #[cls,x,y,z,h,w,l,r,prob]
                        bbox_marker.pose.position.x = float(bbox[1])
                        bbox_marker.pose.position.y = float(bbox[2])
                        #I don't konw why this is so wrong...
                        bbox_marker.pose.position.z = float(bbox[3])-z_offset
                        heading = float(bbox[7])
                        height  = float(bbox[4])
                        width   = float(bbox[5])
                        length  = float(bbox[6])
                        if height > 20 or width > 20 or length > 20:
                            continue
                        q = tf.transformations.quaternion_from_euler(0, 0, heading)
                        bbox_marker.pose.orientation.x = q[0]
                        bbox_marker.pose.orientation.y = q[1]
                        bbox_marker.pose.orientation.z = q[2]
                        bbox_marker.pose.orientation.w = q[3]
                        bbox_marker.scale.x = length
                        bbox_marker.scale.y = width
                        bbox_marker.scale.z = height
                        bbox_marker.color.r = 0.0
                        bbox_marker.color.g = 1.0
                        bbox_marker.color.b = 0.0
                        bbox_marker.color.a = 0.5
                        marker_arr.markers.append(bbox_marker)

                    marker_publisher.publish(marker_arr)                      
                r.sleep()
