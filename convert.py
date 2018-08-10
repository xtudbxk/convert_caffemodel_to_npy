import caffe
import pickle
import numpy as np
import time
import sys


def convert(prototxt_path,model_path,saved_path,use_pickle=False):
    layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8_voc12']
    print("prototxt_path:%s" % prototxt_path)
    print("prototxt_path:%s" % str(type(prototxt_path)))
    net = caffe.Net(str(prototxt_path),str(model_path),caffe.TEST)
    
    params = {}
    for layer in layers:
        params[layer] = {}
        
        params[layer]["w"] = net.params[layer][0].data
        # change C_out x C_in x k_h x k_w to k_h x k_w x C_in x C_out
        params[layer]["w"] = np.swapaxes(params[layer]["w"],0,2)
        params[layer]["w"] = np.swapaxes(params[layer]["w"],1,3)
        params[layer]["w"] = np.swapaxes(params[layer]["w"],2,3)

        params[layer]["b"] = net.params[layer][1].data
        params[layer]["b"] = np.reshape(params[layer]["b"],[-1])
    
    if use_pickle is True:
        with open(saved_path,"wb") as f:
            pickle.dump(params,f)
    else:
        tmp = np.asarray(params)
        np.save(saved_path,tmp)

if __name__ == "__main__":
    prototxt_path = sys.argv[1]
    model_path = sys.argv[2]
    saved_path = sys.argv[3]
    convert(prototxt_path,model_path,saved_path)
