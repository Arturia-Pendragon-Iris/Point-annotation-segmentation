import torch
import torch.nn as nn
from visualization.view_2D import plot_parallel
from distmap import l1_distance_transform, euclidean_distance_transform


def get_point_uncertainty(in_blank, out_blank):
    in_dis = torch.zeros(in_blank.shape).cuda()
    out_dis = torch.zeros(in_blank.shape).cuda()
    for j in range(in_blank.shape[0]):
        in_dis[j, 0] = euclidean_distance_transform(1 - in_blank[j, 0]) - 1
        in_dis[in_dis < 0] = 0

    for j in range(out_blank.shape[0]):
        out_dis[j, 0] = euclidean_distance_transform(1 - out_blank[j, 0]) - 1
        out_dis[in_dis < 0] = 0

    u_d = torch.exp(-in_dis) - torch.exp(-out_dis)
    return u_d


def get_image_uncertainty(input_tensor):
    if input_tensor.shape[1] == 1:
        temp_ct = input_tensor[:, :1]
    else:
        temp_ct = input_tensor[:, :1] - input_tensor[:, 1:]
    # temp_ct[temp_ct <= 0] = 0
    return 1 - 2 / (1 + torch.exp(60 * temp_ct - 15))


def dice_loss(array_1, array_2):
    epsilon = 0.1
    inter = torch.sum(array_1 * array_2)
    norm = torch.sum(array_1 * array_1) + torch.sum(array_2 * array_2)

    return 1 - 2 * (inter + epsilon) / (norm + epsilon)


def global_ce(pre, ct, in_target, out_target):
    u_d = get_point_uncertainty(in_target, out_target)
    u_i = get_image_uncertainty(ct)
    h = torch.abs((u_d + 1) / 2 * (u_i + 1) / 2)

    # v1 = u_d.cpu().detach().numpy()[0, 0]
    # plot_parallel(
    #     a=v1,
    # )

    loss = -torch.sum(torch.abs(u_d) ** 0.5 * (h * torch.log(pre) + (1 - h) * torch.log(1 - pre)))
    return loss


def ce_loss(pre, h):
    return -torch.mean((h * torch.log(pre) + (1 - h) * torch.log(1 - pre)))


def compute_cosine_similarity(v_1, v_2):
    return torch.dot(v_1, v_2) / (torch.norm(v_1) * torch.norm(v_2))


def rsc_loss(F, hess, in_target, out_target):
    loss_sum = 0
    for j in range(F.shape[0]):
        in_point = torch.where(in_target[j, 0] > 0.5)
        out_point = torch.where(out_target[j, 0] > 0.5)

        k1 = len(in_point[0])
        k2 = len(out_point[0])
        # print(k1, k2)
        if not (k1 > 0 and k2 > 0):
            continue

        # fp_average = F[j, :, in_point[0], in_point[1]].mean(dim=1)
        # fq_average = F[j, :, out_point[0], out_point[1]].mean(dim=1)

        fp_average = F[j, :, in_point[0], in_point[1], in_point[2]].mean(dim=1)
        fq_average = F[j, :, out_point[0], out_point[1], out_point[2]].mean(dim=1)

        sub_inter = 0
        for m1 in range(k1):
            # sub_inter += torch.exp(compute_cosine_similarity(
            #     F[j, :, in_point[0][m1], in_point[1][m1]], fq_average)) / k1
            sub_inter += torch.exp(compute_cosine_similarity(
                F[j, :, in_point[0][m1], in_point[1][m1], in_point[2][m1]], fq_average)) / k1
        for m1 in range(k2):
            # sub_inter += torch.exp(compute_cosine_similarity(
            #     F[j, :, out_point[0][m1], out_point[1][m1]], fp_average)) / k2
            sub_inter += torch.exp(compute_cosine_similarity(
                F[j, :, out_point[0][m1], out_point[1][m1], out_point[2][m1]], fp_average)) / k2

        sub_intra = 0
        for m1 in range(k1):
            for m2 in range(k1):
                if m1 == m2:
                    continue
                # s = 1 / (k1 * (k1 - 1)) * torch.exp(compute_cosine_similarity(
                #     hess[j, :, in_point[0][m1], in_point[1][m1]],
                #     hess[j, :, in_point[0][m2], in_point[1][m2]]))
                # sub_intra += torch.exp(compute_cosine_similarity(
                #     F[j, :, in_point[0][m1], in_point[1][m1]],
                #     F[j, :, in_point[0][m2], in_point[1][m2]])) * s

                s = 1 / (k1 * (k1 - 1)) * torch.exp(compute_cosine_similarity(
                    hess[j, :, in_point[0][m1], in_point[1][m1], in_point[2][m1]],
                    hess[j, :, in_point[0][m2], in_point[1][m2], in_point[2][m2]]))
                sub_intra += torch.exp(compute_cosine_similarity(
                    F[j, :, in_point[0][m1], in_point[1][m1], in_point[2][m1]],
                    F[j, :, in_point[0][m2], in_point[1][m2], in_point[2][m2]])) * s

        for m1 in range(k2):
            for m2 in range(k2):
                if m1 == m2:
                    continue
                # s = 1 / (k2 * (k2 - 1)) * torch.exp(compute_cosine_similarity(
                #     hess[j, :, out_point[0][m1], out_point[1][m1]],
                #     hess[j, :, out_point[0][m2], out_point[1][m2]]))
                # sub_intra += torch.exp(compute_cosine_similarity(
                #     F[j, :, out_point[0][m1], out_point[1][m1]],
                #     F[j, :, out_point[0][m2], out_point[1][m2]])) * s
                s = 1 / (k2 * (k2 - 1)) * torch.exp(compute_cosine_similarity(
                    hess[j, :, in_point[0][m1], in_point[1][m1], in_point[2][m1]],
                    hess[j, :, in_point[0][m2], in_point[1][m2], in_point[2][m2]]))
                sub_intra += torch.exp(compute_cosine_similarity(
                    F[j, :, in_point[0][m1], in_point[1][m1], in_point[2][m1]],
                    F[j, :, in_point[0][m2], in_point[1][m2], in_point[2][m2]])) * s

        loss_sum += - sub_intra / sub_inter

    return loss_sum
