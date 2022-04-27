import torch
from torch import nn
from easydict import EasyDict

from model.base_models.ipm import IPM
from model.encoders.neural_view_encoder import NeuralViewEncoder
from model.structures.pointpillar import PointPillarEncoder
from model.encoders.bev_encoder import BevEncoder
from model.encoders.camera_encoder import CameraEncoder
from tools.model_utils import gen_dx_bx
from model.heads.base_head import BaseHead


class HDMapNet(nn.Module):
    def __init__(self, data_conf, model_conf):
        super(HDMapNet, self).__init__()
        embedded_dim = model_conf.embedded_dim
        direction_dim = data_conf.angle_class
        lidar = model_conf.lidar
        self.cam_channel = model_conf.cam_channel
        self.downsample = model_conf.downsample

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camera_encoder = CameraEncoder(self.cam_channel)
        front_view_size = (data_conf['image_size'][0] // self.downsample, data_conf['image_size'][1] // self.downsample)
        bev_size = (final_H // 5, final_W // 5)
        self.view_fusion = NeuralViewEncoder(front_view_size=front_view_size, bev_size=bev_size)

        res_x = bev_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4 * res_x / final_W]
        ipm_ybound = [-res_x / 2, res_x / 2, 2 * res_x / final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.cam_channel, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.lidar = lidar
        if lidar:
            self.point_pillar = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            self.bev_encoder = BevEncoder(in_channel=self.cam_channel + 128)
        else:
            self.bev_encoder = BevEncoder(in_channel=self.cam_channel)
        self.segmentation_head = BaseHead(data_conf['num_channels'])
        self.instance_head = BaseHead(embedded_dim)
        self.direction_head = BaseHead(direction_dim + 1)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.camera_encoder(x)
        x = x.view(B, N, self.cam_channel, imH // self.downsample, imW // self.downsample)
        return x

    def forward(self, data_dict):
        data_dict = EasyDict(data_dict)
        imgs, trans, rots = data_dict.imgs, data_dict.trans, data_dict.rots
        intrins, post_trans, post_rots = data_dict.intrins, data_dict.post_trans, data_dict.post_rots
        lidar_data, lidar_mask = data_dict.lidar_data, data_dict.lidar_mask
        car_trans, yaw_pitch_roll = data_dict.car_trans, data_dict.yaw_pitch_roll
        x = self.get_cam_feats(imgs)
        x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.point_pillar(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        x2, x1 = self.bev_encoder(topdown)
        x_seg = self.segmentation_head(x2, x1)
        x_ins = self.instance_head(x2, x1)
        x_dir = self.direction_head(x2, x1)
        return x_seg, x_ins, x_dir
