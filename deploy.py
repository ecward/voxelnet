# Try to optimize the trained voxelnet model to be more efficent

#First, let's load the graph from checkpoints and save it as a GraphDef file (needed by later steps)

import tensorflow as tf
import argparse
from model import RPN3D
from config import cfg
from tensorflow.core.protobuf import saver_pb2

#if __name__ == '__main__':

# parser = argparse.ArgumentParser(description='deploy')
# parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
#                          help='set log tag')
# args = parser.parse_args()
# save_model_dir = os.path.join('./save_model', args.tag)

save_pb  = True
freeze   = False
optimize = False

save_model_dir = './save_model/smaller_T'


if save_pb:

    #Okay, why number doesn't show UP....
    #maybe it's because the output doesn't depend on it (i.e. we have chosen the wrong output names!)
    
    #this should do the same thing...
    #python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph=smaller_T.pb --input_checkpoint=save_model/smaller_T/checkpoint-00185601 --output_graph=fr00zen.pb --output_node_names=gpu_0/MiddleAndRPN_/p_pos,gpu_0/MiddleAndRPN_/conv21/Conv2D
    
    tf_graph = tf.Graph().as_default()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                visible_device_list=cfg.GPU_AVAILABLE,
                                allow_growth=True)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={
            "GPU": cfg.GPU_USE_COUNT,
        },
        allow_soft_placement=True,
    )


    sess =  tf.Session(config=config)

    #I changed the names of some stuff after training, hopefully it should still work...
    model = RPN3D(
        cls=cfg.DETECT_OBJ,
        single_batch_size=1,
        is_train=True,
        avail_gpus=cfg.GPU_AVAILABLE.split(',')
    )

    if tf.train.get_checkpoint_state(save_model_dir):
        print("Reading model parameters from %s" % save_model_dir)
        print("sess = ",sess)
        model.saver.restore(  sess, tf.train.latest_checkpoint(save_model_dir))

        #maybe I can re-name the outputs here, to avoid all this crap....
        
        
        #In group_pointcloud.py (FeatureNet) tf.placeholders are defined for inputs
        #the same in rpn.py (MiddleAndRPN)

        #here gpu_0/number exists
        graph_def = sess.graph.as_graph_def()
        #node names and stuff...
        #f = open('nodes.txt','w')
        #for node in graph_def.node:
        #    f.write(str(node))
        #f.close()

        
        tf.train.write_graph(graph_def,'.','smaller_T.pb') #here it also exists....
        
        #Also store using the "OLD" format so that we can actually read it later...
        #saver_ok = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
        #saver_ok.save(sess,'smaller_T.ckpt')

    else:
        print("No saved model!","save_model_dir=",save_model_dir)


#FREEZING
#python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph=smaller_T.pb --input_checkpoint=save_model/smaller_T/checkpoint-00185601 --output_graph=fr00zen.pb --output_node_names=concat_104,concat_105

if optimize or freeze:
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib    

    input_graph_path = "./smaller_T.pb"
    checkpoint_path  = "./smaller_T.ckpt"
    input_saver_def_path = ""
    input_binary = False
    #This is the output of voxelnet
    #seems like the node names automatically get pre-fixed by gpu_0/MiddleAndRPN_

    #First one is probability of it being a car ("tag")    
    #Second one is a ConvMD(2, 768, 14, 1, (1, 1), (0, 0),...)
    # Regression(residual) map, scale = [None, 200/100, 176/120, 14]????
    #in the code it is used as this: deltas: (N, w, l, 14)

    #Conv2D seems like the most reasonable output name... (the operations name, without padding and stuff)
    #output_node_names = "gpu_0/MiddleAndRPN_/p_pos,gpu_0/MiddleAndRPN_/conv21/Conv2D"

    #model.prob_output.name -> concat_105
    #model.delta_output.name -> concat_104
    output_node_names = "concat_104,concat_105"
    input_node_names  = "gpu_0/feature:0,gpu_0/number:0,gpu_0/coordinate:0"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'frozen_smaller_T.pb'
    output_optimized_graph_name = 'optimized_smaller_T.pb'
    clear_devices = True

    if freeze:
        print("Freezing graph...")
        freeze_graph.freeze_graph(input_graph_path,
                                  input_saver_def_path,
                                  input_binary,
                                  checkpoint_path,
                                  output_node_names,
                                  restore_op_name,
                                  filename_tensor_name,
                                  output_frozen_graph_name,
                                  clear_devices,
                                  "")
        print("Done!")


    if optimize:
        # Optimize for inference

        input_graph_def = tf.GraphDef()
        with tf.gfile.Open("fr00zen.pb", "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)


        output_node_names = "concat_104,concat_105"
        #gpu_0/number is not used at all so freezing removes it!
        input_node_names  = "gpu_0/feature,gpu_0/coordinate"

        #python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/optimize_for_inference.py --input=fr00zen.pb --output=fr00zen_opt.pb --frozen_graph=True --input_names=gpu_0/feature,gpu_0/coordinate --output_names=concat_104,concat_105


        #this looks wrong...
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def,
                input_node_names.split(","), # an array of the input node(s)
                output_node_names.split(","), # an array of output nodes
                tf.float32.as_datatype_enum)


        # Save the optimized graph

        f = tf.gfile.FastGFile("fr00zen_opt.pb", "w")
        f.write(output_graph_def.SerializeToString())
