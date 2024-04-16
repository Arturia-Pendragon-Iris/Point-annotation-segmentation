import argparse, os
import torch
import math, random
from scipy.ndimage import zoom
import torch.backends.cudnn as cudnn
import cv2
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.inferers import SliceInferer
import numpy as np
from monai.networks.nets import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


def predict_airway(ct_array):
    model = SwinUNETR(img_size=(192, 192, 128),
                      in_channels=1,
                      out_channels=1,
                      spatial_dims=3,
                      feature_size=24)
    pretrained_model = torch.load("/data/Train_and_Test/airway/airway_epoch_2.pth")
    model.load_state_dict(pretrained_model, strict=True)
    model = model.cuda()
    model.half()
    model.eval()

    input_ct = torch.tensor(ct_array[np.newaxis, np.newaxis]).to(torch.float).half()

    with torch.no_grad():
        pre = sliding_window_inference(inputs=input_ct,
                                       predictor=model,
                                       roi_size=(192, 192, 128),
                                       sw_batch_size=2,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.125,
                                       progress=False,
                                       sw_device="cuda",
                                       device="cpu")

    pre = torch.sigmoid(pre[0]).detach().cpu().numpy()[0, 0]
    airway = np.array(pre > 0.51, "float32")

    return airway


def predict_lung(ct_array, lung=None):
    model = SwinUNETR(img_size=(192, 192, 128),
                      in_channels=1,
                      out_channels=1,
                      spatial_dims=3,
                      feature_size=24)
    pretrained_model = torch.load("/data/Train_and_Test/semantic/lung.pth")
    model.load_state_dict(pretrained_model, strict=True)
    model = model.cuda()
    model.half()
    model.eval()

    input_ct = torch.tensor(ct_array[np.newaxis, np.newaxis]).to(torch.float).half()

    with torch.no_grad():
        pre = sliding_window_inference(inputs=input_ct,
                                       predictor=model,
                                       roi_size=(192, 192, 128),
                                       sw_batch_size=2,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.125,
                                       progress=False,
                                       sw_device="cuda",
                                       device="cpu")

    pre = torch.sigmoid(pre[0]).detach().cpu().numpy()[0, 0]
    lung = np.array(pre > 0.51, "float32")

    return lung


