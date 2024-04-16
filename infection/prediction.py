import argparse, os
import torch
import numpy as np
from monai.inferers import SliceInferer, sliding_window_inference
from infection.model import SegNet_2D, SegNet_3D


def predict_infection_slice(ct_array, batch_size=8):
    model = SegNet_2D(in_ch=1, final_ch=4)
    model = model.to('cuda')
    model.load_state_dict(torch.load("/data/Train_and_Test/infection/wM_2D_3.pth"))
    model.half()
    model.eval()

    input_set = torch.from_numpy(ct_array[np.newaxis, np.newaxis]).to(torch.float)
    input_set = input_set.to('cuda').half()

    with torch.no_grad():
        inferer = SliceInferer(spatial_dim=2,
                               roi_size=(512, 512),
                               sw_batch_size=batch_size,
                               progress=False)
        # torch.sigmoid(model(slice_input)[0])
        result = torch.sigmoid(inferer(inputs=input_set, network=model)[0])
        result = result.detach().cpu().numpy()[0, 0]

    return result


def predict_infection_scan(ct_array, batch_size=2):
    model = SegNet_3D(in_ch=1, final_ch=4)
    model = model.to('cuda')
    model.load_state_dict(torch.load("/data/Train_and_Test/infection/wM_3D_3.pth"))
    model.half()
    model.eval()

    input_set = torch.from_numpy(ct_array[np.newaxis, np.newaxis]).to(torch.float)
    input_set = input_set.to('cuda').half()

    with torch.no_grad():
        inferer = sliding_window_inference(inputs=input_set,
                                           predictor=model,
                                           roi_size=(192, 192, 128),
                                           sw_batch_size=batch_size,
                                           overlap=0.25,
                                           mode="gaussian",
                                           sigma_scale=0.125,
                                           progress=False,
                                           sw_device="cuda",
                                           device="cpu")

        result = torch.sigmoid(inferer(inputs=input_set, network=model)[0])
        result = result.detach().cpu().numpy()[0, 0]

    return result


if __name__ == "__main__":
    ct = np.load("/data/chest_CT/rescaled_ct/xgfy-A000010_2020-03-03.npz")["arr_0"]
    ct = np.clip((ct + 1000) / 1600, 0, 1)
    infection = predict_infection_scan(ct)
