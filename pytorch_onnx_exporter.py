import os
import sys
import torch
import torch.onnx
import importlib
from collections import OrderedDict
from argparse import ArgumentParser
import onnx
import onnx.numpy_helper as numpy_helper
import numpy as np

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, help="model file path")                        
    parser.add_argument('--onnx_file_path', required=True, help='onnx file path to be saved.(ex. path/name.onnx)')
    parser.add_argument('--height', type=int, required=True, help='target Height of RGB image')
    parser.add_argument('--width', type=int, required=True, help='target Width of RGB image')
    parser.add_argument('--checkpoint_path', default="", help='(Optional) checkpoint path to load.')
    parser.add_argument('--opset_version', type=int, default=11, help='(Optional) opset_version (default = 11).')
    args = parser.parse_args()
    return args

def get_model(args):
    print("load model.")
    # add path of model to the system.
    sys.path.append(os.path.dirname(args.model_path))
    # import model path.
    net = importlib.import_module(os.path.basename(args.model_path).split('.')[0])
    
    # "Net" should be changed to real model name in model.py
    # "Net" should have right parameters.
    model = net.Net()

    return model

def get_checkpoint(checkpoint_path):
    extension = checkpoint_path.split(".")[-1]
    if extension != 'tar' and extension != 'pth':
        print("checkpoint path/file is not correct.")
        os.system("exit")

    checkpoint = torch.load(checkpoint_path)
    return checkpoint    

def get_state_dict(checkpoint):
    prefix = 'module.'
    state_dict = OrderedDict(
        [(key[7:], value) if key.startswith(prefix) else (key, value)
        for key, value in checkpoint['model'].items()]
    )
    return state_dict

def compare_two_array(actual, desired, layer_name, rtol=1e-30, atol=0):
    flag = False
    try : 
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        print(layer_name + ": no difference.")
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag

def compare_torch_onnx(args, net):
    
    # ③ Reload the created onnx model and compare the weights of the torch model and onnx model    
    onnx_path = args.onnx_file_path
    onnx_model = onnx.load(onnx_path)

    onnx_layers = dict()
    for layer in onnx_model.graph.initializer:
        onnx_layers[layer.name] = numpy_helper.to_array(layer)

    torch_layers = {}
    for layer_name, layer_value in net.named_modules():
        torch_layers[layer_name] = layer_value   

    onnx_layers_set = set(onnx_layers.keys())
    torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
    filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

    difference_flag = False
    for layer_name in filtered_onnx_layers:
        onnx_layer_name = layer_name
        torch_layer_name = layer_name.replace(".weight", "")
        onnx_weight = onnx_layers[onnx_layer_name]
        torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
        compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
        flag = compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
        difference_flag = True if flag == True else False

    # ④ If the onnx model has different weights from the existing torch model, update the whole and save it anew.
    if difference_flag:
        print("update onnx weight from torch model.")
        for index, layer in enumerate(onnx_model.graph.initializer):
            layer_name = layer.name
            if layer_name in filtered_onnx_layers:
                onnx_layer_name = layer_name
                torch_layer_name = layer_name.replace(".weight", "")
                onnx_weight = onnx_layers[onnx_layer_name]
                torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
                copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
                onnx_model.graph.initializer[index].CopyFrom(copy_tensor)
    
        print("save updated onnx model.")
        onnx_new_path = os.path.dirname(os.path.abspath(onnx_path)) + os.sep + "updated_" + os.path.basename(onnx_path)
        onnx.save(onnx_model, onnx_new_path) 
    
    # ⑤ Finally, load the saved onnx model, add shape information, and then save it again.
    if difference_flag:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_new_path)), onnx_new_path)
    else:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

def main(args):

    # ① Load the deep learning network you want to use and set it to evaluation mode.
    model = get_model(args)

    if os.path.isfile(args.checkpoint_path):
        print(">>> load weight.")
        checkpoint = get_checkpoint(args.checkpoint_path)
        state_dict = get_state_dict(checkpoint)
        
        print(">>> apply weight into model.")
        model.load_state_dict(state_dict)
    else:
        print(">>> without pre-trained.")
    model.eval()

    # ② Create the onnx model using the torch model.
    dummy_data = torch.empty(1, 3, args.height, args.width)
    print(">>> onnx export start.")
    torch.onnx.export(model, dummy_data, args.onnx_file_path, input_names=["input"], output_names=["output"], opset_version=args.opset_version)
    print(">>> onnx export end.")
    
    compare_torch_onnx(args, model)

if __name__ == '__main__':

    args = get_arguments()
    main(args)
