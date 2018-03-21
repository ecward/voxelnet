import glob
import argparse
import os
import time
import tensorflow as tf

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
#each process handles its own output
FILESIZE = 100000
POINTCLOUD_FILE = "pc_input.p"
BBOX_FILE       = "bbox_output.p"

model_pb = "fr00zen_opt.pb"

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

    anchors = cal_anchors()    
    
    #inter-process comms.
    f = open(BBOX_FILE,"wb")
    f.write(FILESIZE*b'\0')
    f.close()

    last_idx = None
    
    with open(POINTCLOUD_FILE,"rb") as pc_f:
        with open(BBOX_FILE,"r+b") as bbox_f:

            t0 = time.time()

            idx    = None
            pcloud = None
            with contextlib.closing(mmap.mmap(pc_f.fileno(),0,access=mmap.ACCESS_READ)) as m:
                idx,pcloud  = pickle.load(m)

            if pcloud is not None: #TODO more input checking
                t0_pred = time.time()
                print("raw lidar",np.shape(pcloud))
                voxel = process_pointcloud(pcloud)
                _, per_vox_feature, per_vox_number, per_vox_coordinate = build_input(voxel)
                print("input sizes: feature = ",np.shape(per_vox_feature)," coord = ",np.shape(per_vox_coordinate))
                input_feed[input_feature]    = per_vox_feature
                input_feed[input_coordinate] = per_vox_coordinate

                output_feed = [output_prob, output_delta]

                probs, deltas = sess.run(output_feed, input_feed)
                
                batch_boxes3d = delta_to_boxes3d(deltas, anchors, coordinate='lidar')
                batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
                batch_probs = probs.reshape((1, -1))

                # NMS
                ret_box3d = []
                ret_score = []

                # remove box with low score
                ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
                tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
                tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
                tmp_scores = batch_probs[batch_id, ind]
                
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
                    
                labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                print("predict runtime = ",time.time()-t0_pred)

                #output data to file
                with contextlib.closing(mmap.mmap(bbox_f.fileno(),0,access=mmap.ACCESS_WRITE)) as m:
                    m.seek(0) #rewind
                    pickle.dump(labels,m)
                    m.flush()
