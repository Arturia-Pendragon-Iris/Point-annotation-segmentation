import glob
import os
from torchvision.transforms import transforms
from distance_transfer import perform_distance_trans
import random
import numpy as np
from skimage.measure import label, regionprops
from torch.utils.data.dataset import Dataset
import torch
from hessian_matrix.hessian_2d import compute_hessian as h_2d
from hessian_matrix.hessian_3d import compute_hessian as h_3d
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandHistogramShift,
    Rand2DElastic,
    RandGaussianSharpen,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAffine)


k = np.random.uniform(low=0, high=0.025)
train_transforms = Compose(
    [
     RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(np.pi/10, np.pi/10, np.pi/10),
        scale_range=(-0.1, 0.1)),
     RandFlip(prob=0.2),
     RandRotate90(prob=0.2),
     RandHistogramShift(num_control_points=10, prob=0.2),
     RandGaussianNoise(prob=0.2, mean=0, std=k),
    ]
)


def random_point_from_domains(mask):
    # Label connected regions
    labeled_image = label(mask)
    regions = regionprops(labeled_image)

    # List to store the coordinates of the random points
    random_points = []

    # Iterate over all detected regions
    for region in regions:
        # Extract the coordinates of all the points in the region
        coordinates = list(region.coords)
        if len(coordinates) < 50:
            continue
        # Select a random point from the coordinates
        random_point = random.choice(coordinates)

        # Append the selected point to the list
        random_points.append(random_point)

    return list(random_points)


def random_points(mask, num_points=5):
    # Find the indices of all pixels with the given value
    y_indices, x_indices = np.where(mask > 0.5)
    coordinates = np.column_stack((y_indices, x_indices))
    if len(coordinates) < num_points:
        in_target = []
    else:
        in_target = coordinates[np.random.choice(len(coordinates), size=num_points, replace=False)]

    y_indices, x_indices = np.where(mask < 0.05)
    coordinates = np.column_stack((y_indices, x_indices))
    out_target = coordinates[np.random.choice(len(coordinates), size=num_points, replace=False)]

    return list(in_target), list(out_target)


def get_hessian(ct):
    if len(ct.shape) == 2:
        l1, l2 = h_2d(ct)
        return np.stack((l1, l2), axis=0)
    else:
        l1, l2, l3 = h_3d(ct)
        return np.stack((l1, l2, l3), axis=0)


class TrainSetLoader_2D(Dataset):
    def __init__(self, device):
        super(TrainSetLoader_2D, self).__init__()
        # self.dataset_dir = dataset_dir
        self.file_list = []
        for filename in os.listdir("/data/Train_and_Test/infection_new/xgfy"):
            self.file_list.append(
                os.path.join("/data/Train_and_Test/infection_new/xgfy", filename))

        print(len(self.file_list))
        self.device = device

    def __getitem__(self, index):
        # np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))["arr_0"]
        # np_array = np.clip(np_array, 0, 1)
        # np_array = train_transforms(np_array)
        # gt = np.array(np_array[3] > 0.5, "float32")
        # domain_point = random_point_from_domains(gt)

        # np_array = np.load(self.file_list[index])["arr_0"].item()
        # ct = np.clip(np_array["ct"], 0, 1)
        # gt = np.array(np_array["gt"] > 0.5, "float32")

        np_array = np.load(self.file_list[index])["arr_0"]
        ct = np.clip(np_array[2], 0, 1)
        gt = np.array(np_array[3] > 0.5, "float32")

        in_target = np_array["in_target"]
        out_target = np_array["out_target"]
        # in_target, out_target = random_points(gt, num_points=np.random.randint(low=3, high=6))

        in_blank = np.zeros(ct.shape)
        out_blank = np.zeros(ct.shape)

        for i in range(len(in_target)):
            in_blank[in_target[i][0], in_target[i][1]] = 1
        for i in range(len(out_target)):
            out_blank[out_target[i][0], out_target[i][1]] = 1

        hessian = get_hessian(ct)

        # u_d = get_point_uncertainty(in_target, out_target)
        # u_i = get_image_uncertainty(ct)
        # h = np.abs((u_d + 1) / 2 * (u_i + 1) / 2)

        ct = torch.tensor(ct[np.newaxis]).to(torch.float).to(self.device)
        gt = torch.tensor(gt[np.newaxis]).to(torch.float).to(self.device)
        in_blank = torch.tensor(in_blank[np.newaxis]).to(torch.float).to(self.device)
        out_blank = torch.tensor(out_blank[np.newaxis]).to(torch.float).to(self.device)
        hessian = torch.tensor(hessian).to(torch.float).to(self.device)

        return ct, gt, in_blank, out_blank, hessian

    def __len__(self):
        # print(len(self.dataset_dir))
        return len(self.file_list)

class TrainSetLoader_3D(Dataset):
    def __init__(self, dataset_dir, device):
        super(TrainSetLoader_3D, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = []
        for filename in os.listdir("/data/Train_and_Test/infection_3D"):
            self.file_list.append(
                os.path.join("/data/Train_and_Test/infection_3D", filename))

        print(len(self.file_list))
        self.device = device

    def __getitem__(self, index):
        np_array = np.load(os.path.join(self.dataset_dir, self.file_list[index]))["arr_0"].item()
        ct = np.clip(np_array["ct"], 0, 1)
        x_index = np.random.randint(low=0, high=ct.shape[0] - 192)
        y_index = np.random.randint(low=0, high=ct.shape[1] - 192)
        z_index = np.random.randint(low=0, high=ct.shape[2] - 128)

        ct = ct[x_index:x_index + 192, y_index:y_index + 192, z_index:z_index + 128]
        gt = np_array["gt"][x_index:x_index + 192, y_index:y_index + 192, z_index:z_index + 128]

        in_target = np_array["in_target"]
        out_target = np_array["out_target"]

        for i in range(len(in_target)):
            if in_target[i][0] > x_index or in_target[i][0] < x_index + 192:
                in_target.pop(i)
            if in_target[i][1] > y_index or in_target[i][1] < y_index + 192:
                in_target.pop(i)
            if in_target[i][2] > z_index or in_target[i][2] < z_index + 128:
                in_target.pop(i)

        for i in range(len(out_target)):
            if out_target[i][0] > x_index or out_target[i][0] < x_index + 192:
                out_target.pop(i)
            if out_target[i][1] > y_index or out_target[i][1] < y_index + 192:
                out_target.pop(i)
            if out_target[i][2] > z_index or out_target[i][2] < z_index + 128:
                out_target.pop(i)

        in_blank = np.zeros(ct.shape)
        out_blank = np.zeros(ct.shape)

        for i in range(len(in_target)):
            in_blank[in_target[i][0] - x_index, in_target[i][1] - y_index, in_target[i][2] - z_index] = 1
        for i in range(len(out_target)):
            out_blank[out_target[i][0] - x_index, out_target[i][1] - y_index, out_target[i][2] - z_index] = 1

        hessian = get_hessian(ct)

        ct = torch.tensor(ct[np.newaxis]).to(torch.float).to(self.device)
        gt = torch.tensor(gt[np.newaxis]).to(torch.float).to(self.device)
        in_blank = torch.tensor(in_blank[np.newaxis]).to(torch.float).to(self.device)
        out_blank = torch.tensor(out_blank[np.newaxis]).to(torch.float).to(self.device)
        hessian = torch.tensor(hessian).to(torch.float).to(self.device)

        return ct, gt, in_blank, out_blank, hessian

    def __len__(self):
        # print(len(self.dataset_dir))
        return len(self.file_list)




