import numpy as np
import open3d as o3d
import scipy
import scipy.optimize
import torch

'''
Distance metric implementations from RoboCook: https://github.com/hshi74/robocook/blob/main/utils/loss.py
'''

def chamfer(x, y, pkg="numpy"):
    if pkg == "numpy":
        # numpy implementation
        x = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        dis = np.linalg.norm(x - y, 2, axis=2)
        dis_xy = np.mean(np.min(dis, axis=1))  # dis_xy: mean over N
        dis_yx = np.mean(np.min(dis, axis=0))  # dis_yx: mean over M
    else:
        # torch implementation
        x = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

    return dis_xy + dis_yx

def emd(x, y, pkg="numpy"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")
        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd

def hausdorff(x, y, pkg="numpy"):
    if pkg == "numpy":
        # numpy implementation
        x = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        dis = np.linalg.norm(x - y, 2, axis=2)
        dis_xy = np.max(np.min(dis, axis=1))  # dis_xy: mean over N
        dis_yx = np.max(np.min(dis, axis=0))  # dis_yx: mean over M
    else:
        # torch implementation
        x = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.max(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

    return dis_xy + dis_yx