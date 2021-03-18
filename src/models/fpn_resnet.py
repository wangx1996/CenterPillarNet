"""
# ---------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# Refer from: https://github.com/xingyizhou/CenterNet

# Modifier: Nguyen Mau Dung (2020.08.09)
# ------------------------------------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models.pointpillars import PFNLayer, PillarFeatureNet, PointPillarsScatter
from models import middle

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv,
                 output_shape,
                 voxel_generator,
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 vfe_num_filters=[32, 128],
                 num_input_features=4,
                 with_distance=False,
                 use_norm = True,
                 **kwargs):

        middle_class_name = "PointPillarsScatter",
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads
        self.voxel_generator = voxel_generator
        self.input_h = 608
        self.input_w = 608
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

        self.voxel_feature_extractor = PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )

        self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                            num_input_features=vfe_num_filters[-1])

        fpn_channels = [256, 128, 64]
        for fpn_idx, fpn_c in enumerate(fpn_channels):
            for head in sorted(self.heads):
                num_output = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
                else:
                    fc = nn.Conv2d(in_channels=fpn_c, out_channels=num_output, kernel_size=1, stride=1, padding=0)

                self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, voxels, coors, num_points):

        #_, _, input_h, input_w = x.size()
        hm_h, hm_w = self.input_h // 4, self.input_w // 4

        batch_size_dev = 1

        if len(num_points.shape) == 2:  # multi-gpu
            num_voxel_per_batch = voxels.cpu().numpy().reshape(
                -1)
            voxel_list = []
            num_points_list = []
            coors_list = []
            for i, num_voxel in enumerate(num_voxel_per_batch):
                #print('{}, {}'.format(i, num_voxel))
                voxel_list.append(voxels[i, :int(num_voxel)])
                num_points_list.append(num_points[i, :num_voxel])
                coors_list.append(coors[i, :num_voxel])
            voxels = torch.cat(voxel_list, dim=0)
            num_points = torch.cat(num_points_list, dim=0)
            coors = torch.cat(coors_list, dim=0)
        #print('startLLLLLLLLLL')

        #print('voxel :{}'.format(voxels.shape))
        #print('numpoints:{}'.format(num_points.shape))

        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size_dev)

        #print(spatial_features.shape)

        spatial_features = self.conv1(spatial_features)
        #print('spatial_features {}'.format(spatial_features.shape))
        #print('spatial_features device{}'.format(spatial_features.device))


        spatial_features = self.bn1(spatial_features)
        spatial_features = self.relu(spatial_features)
        spatial_features = self.maxpool(spatial_features)
        #print('spatial_features {}'.format(spatial_features.shape))

        out_layer1 = self.layer1(spatial_features)
        #print('out_layer1 {}'.format(out_layer1.shape))


        out_layer2 = self.layer2(out_layer1)

        #print('out_layer2 {}'.format(out_layer2.shape))


        out_layer3 = self.layer3(out_layer2)

        #print('out_layer3 {}'.format(out_layer3.shape))


        out_layer4 = self.layer4(out_layer3)
        #print('out_layer4 {}'.format(out_layer4.shape))


        # up_level1: torch.Size([b, 512, 14, 14])
        up_level1 = F.interpolate(out_layer4, scale_factor=2, mode='bilinear', align_corners=True)
        #print('up_level1 {}'.format(up_level1.shape))


        concat_level1 = torch.cat((up_level1, out_layer3), dim=1)
        # up_level2: torch.Size([b, 256, 28, 28])
        up_level2 = F.interpolate(self.conv_up_level1(concat_level1), scale_factor=2, mode='bilinear',
                                  align_corners=True)
        #print('up_level2 {}'.format(up_level2.shape))

        concat_level2 = torch.cat((up_level2, out_layer2), dim=1)
        # up_level3: torch.Size([b, 128, 56, 56]),
        up_level3 = F.interpolate(self.conv_up_level2(concat_level2), scale_factor=2, mode='bilinear',
                                  align_corners=True)
        #print('up_level3 {}'.format(up_level3.shape))


        # up_level4: torch.Size([b, 64, 56, 56])
        up_level4 = self.conv_up_level3(torch.cat((up_level3, out_layer1), dim=1))

        from collections import  namedtuple

        ret = {}
        for head in self.heads:
            temp_outs = []
            for fpn_idx, fdn_input in enumerate([up_level2, up_level3, up_level4]):
                fpn_out = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                # Make sure the added features having same size of heatmap output
                if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
                    fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
                temp_outs.append(fpn_out)
            # Take the softmax in the keypoint feature pyramid network
            final_out = self.apply_kfpn(temp_outs)

            ret[head] = final_out

        data_name_tuple = namedtuple('data_name_tuple', ret)
        ret = data_name_tuple(**ret)

        return ret

    def apply_kfpn(self, outs):
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        return ret_outs

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # TODO: Check initial weights for head later
            for fpn_idx in [0, 1, 2]:  # 3 FPN layers
                for head in self.heads:
                    final_layer = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))
                    for i, m in enumerate(final_layer.modules()):
                        if isinstance(m, nn.Conv2d):
                            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                            # print('=> init {}.bias as 0'.format(name))
                            if m.weight.shape[0] == self.heads[head]:
                                if 'hm' in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            model_dict = self.state_dict()
            conv = nn.Conv2d(128, 64, kernel_size=7, stride=1,
                             padding=3, bias=False)
            torch.nn.init.xavier_uniform(conv.weight)
            pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict and k != 'base_layer.0.weight'}
            pretrained_dict['conv1.weight'] = conv.weight
            self.load_state_dict(pretrained_dict, strict=False)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(voxel_generator, num_layers, heads, head_conv, imagenet_pretrained):
    block_class, layers = resnet_spec[num_layers]
    vfe_num_filters = list([64])
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    model = PoseResNet(block_class, layers, heads, head_conv=head_conv,output_shape=dense_shape, voxel_generator=voxel_generator)
    model.init_weights(num_layers, pretrained=imagenet_pretrained)
    ckpt_path = '/media/wx/File/data/pretrained_models_v1.5/car_fhd/adam_optimizer-74280.ckpt'
    #model_dict = model.state_dict()
    state_dict = torch.load(ckpt_path)
    #pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    #model_dict.update(pretrained_dict)
    model.load_state_dict(state_dict, strict = False)
    return model
