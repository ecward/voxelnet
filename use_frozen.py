#!/usr/bin/env python


import glob
import argparse
import os
import time
import tensorflow as tf

#from model import RPN3D
from config import cfg
from utils import *
from utils.kitti_loader import iterate_data, sample_test_data

from tensorflow.core.framework import graph_pb2

#Let's try to load the frozen model and use it...
#model_pb = "fr00zen.pb"#"frozen_smaller_T.pb"
model_pb = "fr00zen_opt.pb"

#Quantized is 100x slower...
#model_pb = "quantized_graph.pb"

#let's see if this makes stuff take a lot of time.. nope..
do_post = True

f =  open(model_pb,'rb')

print("Reading graph from string...")
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
print("Done!")
f.close()

#let's see that we have our output nodes...
#delta_output,prob_output...
output_node_names = "concat_104,concat_105"
output_node_names = output_node_names.split(",")
input_node_names = ["gpu_0/feature","gpu_0/number","gpu_0/coordinate"]

#Number is not used....
#print("Input nodes")
#for i_node_name in input_node_names:
#    print(i_node_name,":")
#    print([n for n in graph_def.node if n.name.find(i_node_name) != -1])

# print("Output nodes")
# for o_node_name in output_node_names:
#     print(o_node_name,":")
#     print([n for n in graph_def.node if n.name.find(o_node_name) != -1])



#Maybe turn off allow_growth?
frac=0.8
#per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac,
                            visible_device_list=cfg.GPU_AVAILABLE,
                            allow_growth=False)

#what is soft placement? Use CPU verison if we don't have GPU version....
config = tf.ConfigProto(
    gpu_options=gpu_options,
    device_count={
        "GPU": cfg.GPU_USE_COUNT,
    },
    #allow_soft_placement=True,
    allow_soft_placement=False,
)

sess =  tf.Session(config=config)
tf.import_graph_def(graph_def, name='')


input_feature    = sess.graph.get_tensor_by_name("gpu_0/feature:0")
input_coordinate = sess.graph.get_tensor_by_name("gpu_0/coordinate:0")
print(str(input_feature))
print(str(input_coordinate))

#model.prob_output.name -> concat_105
#model.delta_output.name -> concat_104
output_prob  = sess.graph.get_tensor_by_name("concat_105:0")
output_delta = sess.graph.get_tensor_by_name("concat_104:0")


val_dir =  os.path.join(cfg.DATA_DIR, 'validation')

anchors = cal_anchors()
#Non-max-suppression (seems like it is done on the cpu anyway)
#postprocessing on GPU?
if do_post:
    tf_boxes2d        = tf.placeholder(tf.float32, [None, 4])
    tf_boxes2d_scores = tf.placeholder(tf.float32, [None])
    #let's try running this on cpu..
    #with tf.device('/gpu:0'):
    with tf.device('/cpu:0'):
        tf_box2d_ind_after_nms = tf.image.non_max_suppression(
            tf_boxes2d, tf_boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)


#okay, let's get some data the same way as before...
for batch in iterate_data(val_dir, shuffle=False, aug=False, is_testset=False, batch_size=1, multi_gpu_sum=1):
    for step in range(10):
        t0 = time.time()
        #tags, results = model.predict_step(sess, batch, summary=False, vis=False)
        tag            = batch[0]    
        label          = batch[1]
        vox_feature    = batch[2]
        vox_number     = batch[3]
        vox_coordinate = batch[4]
        img            = batch[5]
        lidar          = batch[6]
        input_feed = {}
        input_feed[input_feature]    = vox_feature[0]
        input_feed[input_coordinate] = vox_coordinate[0]
        #Best bet to reduce memory usage (reduce the number!)
        #process_pointcloud...
        
        print("input sizes: feature = ",np.shape(vox_feature[0])," coord = ",np.shape(vox_coordinate[0]))
        output_feed = [output_prob, output_delta]

        probs, deltas = sess.run(output_feed, input_feed)
        
        batch_boxes3d = delta_to_boxes3d(deltas, anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = probs.reshape((1, -1))

        if do_post:
            # NMS
            ret_box3d = []
            ret_score = []
            for batch_id in range(1):
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
        
        tags = tag
        print("predict runtime = ",time.time()-t0,"s")
        
    #print("ret_box3d_score = ",ret_box3d_score)
    if do_post:
        for tag, result in zip(tags, ret_box3d_score):
            of_path = os.path.join('fr00zen_data'+ tag + '.txt')
            with open(of_path, 'w+') as f:
                labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                for line in labels:
                    f.write(line)
                print('write out {} objects to {}'.format(len(labels), tag))
