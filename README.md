## **What is pytorch_onnx_exporter ?**

<br>

- This code is to **export onnx file from computer vision pytorch model.** Input type is related to single color image (Channel=3, Height, Width).

<br>

## **What shoud I change ?**

<br>

- ① You should change `model = net.Net()` in `get_model` function. `Net()` means model's representative class name (ex. ResNet(), DenseNet() ...) in your model.py and `Net()` has necessary parameters you set.
- ② If you don't use input as single image, please revise `dummy_data = torch.empty(1, 3, args.height, args.width)` as your model's input type.
- ③ If your model has multiple outputs, please revise `output_names` of `torch.onnx.export`.

<br>

## **What does this code operate ?**

<br>

- ① Load the deep learning network you want to use and set it to evaluation mode.
- ② Create the onnx model using the torch model.
- ③ Reload the created onnx model and compare the weights of the torch model and onnx model    
- ④ If the onnx model has different weights from the existing torch model, update the whole and save it anew.
- ⑤ Finally, load the saved onnx model, add shape information, and then save it again.

<br>

## **Dependencies**

<br>

```
pip install torch 
pip install onnx
pip install numpy
```

<br>

## **What arguments do it needs ?**

<br>

- Arguments below are required and optional. You can use like below.

<br>

```
- `python pytorch_noonx_exporter \
    --model_path=/path/.../to/.../model.py \
    --onnx_file_path=/path/.../to/.../output.onnx \
    --height=512 \
    --width=1024 \
    --checkpoint_path=/path/.../to/.../checkpoint.pth \
    --opset_version=11
```

<br>

- Arguments list are below.

<br>

```python
parser.add_argument('--model_path', required=True, help="model file path")                        
parser.add_argument('--onnx_file_path', required=True, help='onnx file path to be saved.(ex. path/name.onnx)')
parser.add_argument('--height', type=int, required=True, help='target Height of RGB image')
parser.add_argument('--width', type=int, required=True, help='target Width of RGB image')
parser.add_argument('--checkpoint_path', default="", help='(Optional) checkpoint path to load.')
parser.add_argument('--opset_version', type=int, default=11, help='(Optional) opset_version (default = 11).')
```