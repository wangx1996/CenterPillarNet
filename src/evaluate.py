"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for training
# Modified: Wang Xu
# email: wangxubit@foxmail.com
"""

import argparse
import os
import time
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from config import kitti_config as cnf
import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict
import cv2
sys.path.append('./')

from data_process.kitti_dataloader import create_val_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processingv2, get_batch_statistics_rotated_bbox, ap_per_class, load_classes
from spconv.utils import VoxelGeneratorV2
from utils.torch_utils import _sigmoid

def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            metadatas, targets= batch_data

            batch_size = len(metadatas['img_path'])

            voxelinput = metadatas['voxels']
            coorinput = metadatas['coors']
            numinput = metadatas['num_points']

            dtype = torch.float32
            voxelinputr = torch.tensor(
                voxelinput, dtype=torch.float32, device=configs.device).to(dtype)

            coorinputr = torch.tensor(
                coorinput, dtype=torch.int32, device=configs.device)

            numinputr = torch.tensor(
                numinput, dtype=torch.int32, device=configs.device)
            t1 = time_synchronized()
            outputs =  model(voxelinputr, coorinputr, numinputr)
            outputs = outputs._asdict()

            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)


            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processingv2(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)

            for sample_i in range(len(detections)):
                # print(output.shape)
                num = targets['count'][sample_i]
                # print(targets['batch'][sample_i][:num].shape)
                target = targets['batch'][sample_i][:num]
                #print(target[:, 8].tolist())
                labels += target[:, 8].tolist()


            sample_metrics += get_batch_statistics_rotated_bbox(detections, targets, iou_threshold=configs.iou_thresh)

            t2 = time_synchronized()

            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='/media/wx/File/kittidatabase/classes_names_pillar.txt',
                        metavar='PATH', help='The class names of objects in the task')
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
    parser.add_argument('--batch_size', type=int, default=4,
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

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

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
    configs.root_dir = '../'
    configs.dataset_dir = '/media/wx/File/kittidatabase'

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs



if __name__ == '__main__':
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.classnames_infor_path)
    print(configs.iou_thresh)

    voxel_generator = VoxelGeneratorV2(
            voxel_size=list(configs.voxel_size),
            point_cloud_range = list(configs.point_cloud_range),
            max_num_points= configs.max_number_of_points_per_voxel,
            max_voxels=20000
            )

    model = create_model(configs, voxel_generator)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    print('Create the validation dataloader')
    val_dataloader = create_val_dataloader(configs, voxel_generator)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {}\n".format(AP.mean()))
