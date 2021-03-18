"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Wang Xu
# email: wangxubit@foxmail.com
# DoC: 2021.03.10
-----------------------------------------------------------------------------------
# Description: ROS Test script
"""
#！/usr/bin/env python3
import argparse
import sys
import os
import time
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from easydict import EasyDict as edict
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing
from utils.misc import time_synchronized
from spconv.utils import VoxelGeneratorV2
from utils.visualization_utils import merge_rgb_to_bev


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (432, 432)
    configs.hm_size = (216, 216)
    configs.down_ratio = 2
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 256
    configs.num_classes = 1
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos
    configs.voxel_size = [0.16, 0.16, 4]
    configs.point_cloud_range = [0, -34.56, -2.73, 69.12, 34.56, 1.27]
    configs.max_number_of_points_per_voxel = 100

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = '/media/wx/File/data/kittidata'

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs

from data_process.kitti_data_utils import get_filtered_lidar
from utils.evaluation_utils import decode,draw_predictions
from utils.torch_utils import _sigmoid
from data_process.kitti_bev_utils import makeBEVMap

class Detection:
    def __init__(self, n, configs, voxel_generator):
        self.node = n
        #self.pub = rospy.Publisher('tracking_result', Object_with_id, queue_size=10)
        self.voxel_generator = voxel_generator
        self.model = create_model(configs, self.voxel_generator)
        #self.model.print_network()
        print('\n\n' + '-*=' * 30 + '\n\n')
        assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
        self.model.load_state_dict(torch.load(configs.pretrained_path,map_location='cuda:0'))
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
        self.model = self.model.to(device=configs.device)
        self.model.eval()
        rospy.Subscriber("kitti/velo/pointcloud", PointCloud2, self.callback) # your cloud topic name
        rospy.spin()

    def callback(self,data):
        rospy.loginfo("detection")
        with torch.no_grad():
            gen = point_cloud2.read_points(data)
            #print(type(gen))
            cloudata = []
            for idx, p in enumerate(gen):
                data = np.array([p[0], p[1], p[2], p[3]])
                data = data.reshape((1,4))
                cloudata.append(data)

            lidarData = np.concatenate([x for x in cloudata], axis=0)

            lidarData = get_filtered_lidar(lidarData, cnf.boundary)
            res = self.voxel_generator.generate(
                lidarData, 20000)



            coorinput = np.pad(
                res["coordinates"], ((0, 0), (1, 0)), mode='constant', constant_values=0)
            voxelinput = res["voxels"]
            numinput = res["num_points_per_voxel"]


            dtype = torch.float32
            voxelinputr = torch.tensor(
                voxelinput, dtype=torch.float32, device=configs.device).to(dtype)

            coorinputr = torch.tensor(
                coorinput, dtype=torch.int32, device=configs.device)

            numinputr = torch.tensor(
                numinput, dtype=torch.int32, device=configs.device)

            outputs =  self.model(voxelinputr, coorinputr, numinputr)
            outputs = outputs._asdict()
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)

            detections = detections[0]

            bev_map = np.ones((432, 432, 3), dtype=np.uint8)
            bev_map = bev_map * 0.5

            bev_map = makeBEVMap(lidarData,cnf.boundary)
            bev_map = bev_map.transpose((1,2,0))*255


            #bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            out_img = bev_map
            cv2.imshow('test-img', out_img)
            cv2.waitKey(1)

if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing
    node = rospy.init_node("detection_listener",anonymous=True) #rosnode
    voxel_generator = VoxelGeneratorV2(
            voxel_size=list(configs.voxel_size),
            point_cloud_range = list(configs.point_cloud_range),
            max_num_points= configs.max_number_of_points_per_voxel,
            max_voxels=20000
            )
    Detection(node, configs, voxel_generator)
    rospy.spin()
