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
from utils.evaluation_utils import decode, post_processingv2, \
    get_batch_statistics_rotated_bbox, ap_per_class, \
    load_classes, convert_det_to_real_values_v2
from utils.visualization_utils import project_to_image, compute_box_3d, draw_box_3d
from data_process.transformation import lidar_to_camera_box
from spconv.utils import VoxelGeneratorV2
from utils.torch_utils import _sigmoid
from data_process.kitti_data_utils import Calibration
import mayavi.mlab
import config.kitti_config as cnf
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

V2C= np.array([7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04,
                           -4.069766000000e-03, 1.480249000000e-02, 7.280733000000e-04,
                           -9.998902000000e-01, -7.631618000000e-02, 9.998621000000e-01,
                           7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01])
V2C = np.reshape(V2C, [3, 4])

C2V = inverse_rigid_trans(V2C)

R0 = np.array([9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03,  -9.869795000000e-03,
                    9.999421000000e-01, -4.278459000000e-03, 7.402527000000e-03, 4.351614000000e-03,
                    9.999631000000e-01])
R0 = np.reshape(R0, [3, 3])


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def project_ref_to_velo(pts_3d_ref):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))


def project_rect_to_ref(pts_3d_rect):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))


def project_rect_to_velo(pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect)
    return project_ref_to_velo(pts_3d_ref)

def rotz(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,  c,  0],
                     [0, 0,  1]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def draw_gt_boxes3d(gt_boxes3d, score,fig,  color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None, ):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        #if draw_text: mayavi.mlab.text3d(b[4,0], b[4,1], b[4,2], 'car'+"{:.2f}".format(float(score)), scale=text_scale, color=(1,1,1), figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def show3dlidar(pointpaht, detections,V2C, R0, P2):
    pointcloud = np.fromfile(pointpaht, dtype=np.float32).reshape(-1, 4)
    x = pointcloud[:, 0]  # x position of point
    xmin = np.amin(x, axis=0)
    xmax = np.amax(x, axis=0 )
    y = pointcloud[:, 1]  # y position of point
    ymin = np.amin(y, axis=0)
    ymax = np.amax(y, axis=0)
    z = pointcloud[:, 2]  # z position of point
    zmin = np.amin(z, axis=0)
    zmax = np.amax(z, axis=0)
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         # 灰度图的伪彩映射
                         colormap='Blues',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    # 绘制原点
    mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere",scale_factor=0.2)

    print(detections.shape)

    detections[:, 1:8] = lidar_to_camera_box(detections[:, 1:8], V2C, R0, P2)

    for i in range(detections.shape[0]):

        h = float(detections[i][4])
        w = float(detections[i][5])
        l = float(detections[i][6])

        x = float(detections[i][1])
        y = float(detections[i][2])
        z = float(detections[i][3])
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2] ;
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h] ;
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
        #print(x_corners)
        #print(detections[i])
        R = roty(float(detections[i][7]))
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        #corners_3d = np.zeros((3,8))
        corners_3d[0, :] = corners_3d[0, :] + x;
        corners_3d[1, :] = corners_3d[1, :] + y;
        corners_3d[2, :] = corners_3d[2, :] + z;
        corners_3d = np.transpose(corners_3d)
        box3d_pts_3d_velo = project_rect_to_velo(corners_3d)
        #x1, y1, z1 = box3d_pts_3d_velo[0, :]
        #x2, y2, z2 = box3d_pts_3d_velo[1, :]
        if detections[i][0] == 1.0:
            draw_gt_boxes3d([box3d_pts_3d_velo],1,color=(1,0,0), fig=fig)
        else:
            draw_gt_boxes3d([box3d_pts_3d_velo], 1, color=(0, 1, 0), fig=fig)

    # 绘制坐标
    '''axes = np.array(
        [[20.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 0.0]],
        dtype=np.float64,
    )
    #x轴
    mayavi.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    #y轴
    mayavi.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    #z轴
    mayavi.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )'''
    mayavi.mlab.show()

def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()

    class_id = {0:'Car', 1:'Pedestrian', 2:'Cyclist'}

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
            img_path = metadatas['img_path'][0]
            #print(img_path)
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))

            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)

            for i in range(configs.batch_size):
                detections[i] = convert_det_to_real_values(detections[i])
                img_path = metadatas['img_path'][i]
                #rint(img_path)
                datap = str.split(img_path,'/')
                filename = str.split(datap[7],'.')
                file_write_obj = open('../result/' + filename[0] + '.txt', 'w')
                lidar_path = '/' + datap[1] + '/' + datap[2] + '/' + datap[3] + '/' + \
                             datap[4] + '/' + datap[5] + '/' + 'velodyne' + '/' + filename[0] + '.bin'
                #print(lidar_path)
                #show3dlidar(lidar_path, detections[i], calib.V2C, calib.R0, calib.P2)
                dets = detections[i]
                if len(dets) >0 :
                    dets[:, 1:] = lidar_to_camera_box(dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                    for box_idx, label in enumerate(dets):
                        location, dim, ry = label[1:4], label[4:7], label[7]
                        if ry < -np.pi:
                            ry = 2*np.pi + ry
                        if ry > np.pi:
                            ry = -2*np.pi + ry
                        corners_3d = compute_box_3d(dim, location, ry)
                        corners_2d = project_to_image(corners_3d, calib.P2)
                        minxy = np.min(corners_2d, axis=0)
                        maxxy = np.max(corners_2d, axis=0)
                        bbox = np.concatenate([minxy, maxxy], axis=0)
                        if bbox[0] < 0 or bbox[2]<0:
                            continue
                        if bbox[1] > 1272 or bbox[3] > 375:
                            continue
                        oblist = ['Car',' ','0.0', ' ', '0', ' ', '-10', ' ','%.2f'%bbox[0], ' ', \
                              '%.2f' %bbox[1], ' ','%.2f'%bbox[2], ' ','%.2f'%bbox[3], ' ','%.2f'%dim[0], ' ','%.2f'%dim[1], ' ','%.2f'%dim[2], ' ', \
                              '%.2f' %location[0], ' ','%.2f'%location[1], ' ','%.2f'%location[2], ' ', '%.2f'%ry, '\n']
                        file_write_obj.writelines(oblist)
                file_write_obj.close()

            '''for sample_i in range(len(detections)):
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
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)'''

    #return precision, recall, AP, f1, ap_class


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
    parser.add_argument('--peak_thresh', type=float, default=0.3)
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
    evaluate_mAP(val_dataloader, model, configs, None)
    '''print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {}\n".format(AP.mean()))'''
