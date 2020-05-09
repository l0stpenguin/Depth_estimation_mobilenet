import os
import glob
import time
from PIL import Image
import numpy as np
import PIL
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import torchvision.models as models
import cv2
from matplotlib import cm
from UtilityTest import DepthDataset
from UtilityTest import ToTensor
from Mobile_model import Model
import onnx
import onnxruntime
from onnx import optimizer, helper, shape_inference

def test_model_accuracy(export_model_name,raw_output, input):    
    ort_session = onnxruntime.InferenceSession(export_model_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)	

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

model = Model().cpu()
model = nn.DataParallel(model)
#load the trained model
model.load_state_dict(torch.load('diml_outdoor_19.pth',map_location=torch.device('cpu')))
model.eval()

# os.mkdir('./generated_img')
# for i,sample_batched1  in enumerate (train_loader):
loc_img="./images"
depth_dataset = DepthDataset(root_dir=loc_img,transform=transforms.Compose([ToTensor()]))
batch_size=1
train_loader=torch.utils.data.DataLoader(depth_dataset, batch_size)
dataiter = iter(train_loader)
images = dataiter.next()
# i = 1
# image = (Image.open("2_image.jpg"))
# transform=transforms.Compose([ToTensor()])
# sample_batched1 = transform({'image': image})
for i,sample_batched1  in enumerate (train_loader):
    image1 = torch.autograd.Variable(sample_batched1['image'].cpu())
    outtt=model(image1 )
    torch.onnx.export(model.module, image1, "depth.onnx", verbose=False, export_params=True, opset_version=10,
    training=False, do_constant_folding=False, keep_initializers_as_inputs=True)
    os.system("python3 -m onnxsim depth.onnx depth_mobilenet.onnx")
    break
    # prediction = model.forward(sample.clone())
    # test_model_accuracy("depth_mobilenet.onnx",outtt, image1)
    # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    
    # onnx_model = onnx.load("depth_10.onnx")
    # optmised = shape_inference.infer_shapes(onnx_model)
    # onnx.save(optmised, 'depth_10_op.onnx')
    # onnx.checker.check_model(onnx_model)
    # graph_output = onnx.helper.printable_graph(onnx_model.graph)
    # with open("graph_output.txt", mode="w") as fout:
    #     fout.write(graph_output)
    x=outtt.detach().cpu().numpy()
    
    # img=x[-1::]
    
    # img=x.reshape(1,x.shape[2],x.shape[3])
    img=x.reshape(240,320)
    print('processing image {0} min {1} max {2}'.format(i,img.min(),img.max()))
    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    plt.imsave('./generated_img/%d_depth.jpg' %i, resized, cmap=cm.gray)

    s_img=sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0)
    plt.imsave('./generated_img/%d_image.jpg' %i, s_img) 