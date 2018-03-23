import glob
import argparse
import os
import time
import tensorflow as tf

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import RPN3D
from config import cfg
from utils import *
from utils.kitti_loader import iterate_data, sample_test_data

from tensorflow.core.framework import graph_pb2

#ugh ros works with python2.7, my tensorflow version is for python3...
#maybe easiest to use tensorflow for python2, but let's try mmap
#and use two processies
import mmap
import contextlib
import numpy as np
import pickle

#two files one for input data to classify (pointcloud as numpy array)
#another with output data (bounding boxes as numpy array)
#commincation using memory mapped files,
#this process creates both files

#10 MB
FILESIZE_PC   = 10000000
#100 kb
FILESIZE_BBOX =   100000

POINTCLOUD_FILE = "pc_input.p"
BBOX_FILE       = "bbox_output.p"

model_pb = "../fr00zen_opt.pb"

USE_FUSED_CLOUDS = True

if __name__ == "__main__":

    #Set up network and tf session
    f =  open(model_pb,'rb')
    print("Reading graph from string...")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print("Done!")
    f.close()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,
                                visible_device_list=cfg.GPU_AVAILABLE,
                                allow_growth=False)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={
            "GPU": cfg.GPU_USE_COUNT,
        },
        allow_soft_placement=False,
    )

    sess =  tf.Session(config=config)
    tf.import_graph_def(graph_def, name='')
    input_feature    = sess.graph.get_tensor_by_name("gpu_0/feature:0")
    input_coordinate = sess.graph.get_tensor_by_name("gpu_0/coordinate:0")
    output_prob  = sess.graph.get_tensor_by_name("concat_105:0")
    output_delta = sess.graph.get_tensor_by_name("concat_104:0")

    tf_boxes2d        = tf.placeholder(tf.float32, [None, 4])
    tf_boxes2d_scores = tf.placeholder(tf.float32, [None])

    with tf.device('/cpu:0'):
        tf_box2d_ind_after_nms = tf.image.non_max_suppression(
            tf_boxes2d, tf_boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)


    #let's not set X_MIN,X_MAX = (0,70.4) but instead (-35.2,35.2)
    if USE_FUSED_CLOUDS:
        cfg.X_MIN = -35.2
        cfg.X_MAX =  35.2
        
    anchors = cal_anchors()    
    
    #inter-process comms.
    f = open(BBOX_FILE,"wb")
    f.write(FILESIZE_BBOX*b'\0')
    f.flush()
    f.close()

    f = open(POINTCLOUD_FILE,"wb")
    f.write(FILESIZE_PC*b'\0')
    f.flush()
    f.close()

    last_idx = None
    
    with open(POINTCLOUD_FILE,"rb") as pc_f:
        with open(BBOX_FILE,"r+b") as bbox_f:

            with mmap.mmap(pc_f.fileno(),0,mmap.MAP_SHARED,mmap.PROT_READ) as buf_pc:
                with mmap.mmap(bbox_f.fileno(),0,mmap.MAP_SHARED,mmap.PROT_WRITE) as buf_bbox:

                    #Main loop
                    while(True):

                        t0 = time.time()

                        idx    = None
                        pcloud = None

                        try:
                            buf_pc.seek(0)
                            idx,pcloud  = pickle.load(buf_pc,encoding='latin1')
                        except Exception as e:
                            print(e)
                            print("Failed to load pointcloud")
                            time.sleep(0.2)

                        if pcloud is not None: #TODO more input checking

                            if last_idx is None:
                                last_idx = idx
                            else:
                                if idx == last_idx:
                                    time.sleep(0.1)
                                    continue #no new data, try to read new data again
                                else:
                                    last_idx = idx

                            t0_pred = time.time()
                            #re-shape pointcloud to what is expected
                            pcloud = pcloud.reshape((-1, 4))
                            print("idx =",idx,"raw lidar",np.shape(pcloud))

                            #shift everything to the left,and up (grid starts at 0,0,0)
                            #also shift forward
                            ofst = np.array([-cfg.X_MIN, cfg.Y_MAX, 3], dtype=np.float32)
                            voxel = process_pointcloud(pcloud,lidar_coord=ofst)
                            _, per_vox_feature, per_vox_number, per_vox_coordinate = build_input([voxel])
                            #convert to np.arrays..
                            per_vox_feature    = np.array([per_vox_feature])
                            per_vox_number     = np.array([per_vox_number])
                            per_vox_coordinate = np.array([per_vox_coordinate])
                            print("input sizes: feature = ",np.shape(per_vox_feature[0])," coord = ",np.shape(per_vox_coordinate[0]))
                            input_feed = {}
                            input_feed[input_feature]    = per_vox_feature[0]
                            input_feed[input_coordinate] = per_vox_coordinate[0]

                            output_feed = [output_prob, output_delta]

                            probs, deltas = sess.run(output_feed, input_feed)

                            batch_boxes3d = delta_to_boxes3d(deltas, anchors, coordinate='lidar')
                            batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
                            batch_probs = probs.reshape((1, -1))

                            # NMS
                            ret_box3d = []
                            ret_score = []

                            # remove box with low score
                            #cfg.RPN_SCORE_THRESH (0.96)
                            ind = np.where(batch_probs[0, :] >= 0.8)[0]
                            tmp_boxes3d = batch_boxes3d[0, ind, ...]
                            tmp_boxes2d = batch_boxes2d[0, ind, ...]
                            tmp_scores = batch_probs[0, ind]

                            # TODO: if possible, use rotate NMS
                            boxes2d = corner_to_standup_box2d(
                                center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
                            ind = sess.run(tf_box2d_ind_after_nms, {
                                tf_boxes2d: boxes2d,
                                tf_boxes2d_scores: tmp_scores
                            })
                            tmp_boxes3d = tmp_boxes3d[ind, ...]
                            tmp_scores = tmp_scores[ind]
                            ret_box3d.append(tmp_boxes3d)
                            ret_score.append(tmp_scores)

                            ret_box3d_score = []
                            for boxes3d, scores in zip(ret_box3d, ret_score):
                                ret_box3d_score.append(np.concatenate([np.tile('Car', len(boxes3d))[:, np.newaxis],
                                                                       boxes3d, scores[:, np.newaxis]], axis=-1))

                            print("ret_box3d_score len = ",len(ret_box3d_score))
                            #let's think about the coordinate system here...
                            
                            for result in ret_box3d_score:

                                #this converts to kitty camera frame...
                                #labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                                labels = result
                                print("detected ",len(labels)," objects")
                            print("predict runtime = ",time.time()-t0_pred)

                            #output data to file
                            result = ret_box3d_score[0] #list of [cls,x,y,z,h,w,l,r]
                            print("Dumpig result at idx=",idx,"=",result)
                            
                            buf_bbox.seek(0)
                            pickle.dump((idx,result),buf_bbox,protocol=2)
                            buf_bbox.flush()
                            print("Done!")
                            # with contextlib.closing(mmap.mmap(bbox_f.fileno(),0,access=mmap.ACCESS_WRITE)) as m:
                            #     m.seek(0) #rewind
                            #     pickle.dump(labels,m)
                            #     m.flush()


                        #End if pcloud is not None
                    #End main loop
