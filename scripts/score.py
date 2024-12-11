# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Test script for audio2pose

import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import torch.types
import wandb
from enum import Enum

from utils import config, logger_tools, other_tools, metric, rotation_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

class TestModel(Enum):
    FRGG = 1
    CAMN = 2
    DIFFSHEG = 3
class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.beat_ori_list = data_tools.joints_list["beat_joints"]
        self.beat_target_list = data_tools.joints_list["spine_neck_141"]
        self.skip_head_bones = False
        self.read_from_out_file = False
        self.test_model = TestModel.FRGG
        if self.test_model == TestModel.FRGG:
            self.out_file_folder_path = "./score_test/FRGG_2468_574/"
            self.out_file_id_prefix = "res_"
            self.remove_tar_first_frame = False #False
        elif self.test_model == TestModel.CAMN:
            self.out_file_folder_path = "./score_test/CaMNEulerAng/"
            self.out_file_id_prefix = "res_"
            self.remove_tar_first_frame = True
        elif self.test_model == TestModel.DIFFSHEG:
            self.out_file_folder_path = "./score_test/DiffSHEG/"
            self.out_file_id_prefix = ""
            self.remove_tar_first_frame = True
        self.out_file_id_suffix = ".bvh"
        self.notes = args.notes
        self.ddp = args.ddp
        self.rank = dist.get_rank()
        self.checkpoint_path = args.root_path+args.out_root_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.root_path+args.out_root_path+"/"+args.name
        self.batch_size = args.batch_size
        self.gpus = len(args.gpus)
        self.trainer_name = args.trainer
        self.best_epochs = {
            'fid_val': [np.inf, 0],
            'rec_val': [np.inf, 0],
                           }
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'), 
        } 
        self.pose_version = args.pose_version
        # data and path
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/{args.pose_rep}_mean.npy")
        self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/{args.pose_rep}_std.npy")
        self.mean_pose_gpu = torch.tensor(self.mean_pose).cuda().float()
        self.std_pose_gpu = torch.tensor(self.std_pose).cuda().float()
        
        # pose
        self.pose_rep = args.pose_rep 
        self.pose_fps = args.pose_fps
        self.output_joint_dims = args.joint_nums * args.output_joint_dims
        self.train_joint_dims = args.train_joint_dims
        self.train_pose_dims = args.joint_nums * args.train_joint_dims
        # audio
        self.audio_rep = args.audio_rep
        self.audio_fps = args.audio_fps
        #self.audio_dims = args.audio_dims
        # facial
        self.facial_rep = args.facial_rep
        self.facial_fps = args.facial_fps
        self.facial_dims = args.facial_dims
        self.pose_length = args.pose_length
        self.stride = args.stride
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        # model para    
        self.pre_frames = args.pre_frames
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_vis_rep}/"
       
        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
        )
        logger.info(f"Init test dataloader success")
        
        if not self.read_from_out_file:
            model_module = __import__(f"models.{args.model}", fromlist=["something"])
            self.model = getattr(model_module, args.g_name)(args)
            other_tools.load_checkpoints(self.model, args.root_path+args.test_ckpt, args.g_name)
            self.model = torch.nn.DataParallel(self.model, args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.model)
                wandb.watch(self.model)
                logger.info(f"init {args.g_name} success")
        
        self.metric_calculator = other_tools.Metric_calculator(args, self.rank)
    def readBVH(self, filepath):
        data_all = []
        x_min = y_min = z_min = 360
        x_max = y_max = z_max = -360
        with open(filepath, "r") as bvh_file:
            for i, line_data in enumerate(bvh_file.readlines()):
                if i < 431:
                    continue
                # one line(i) = one frame
                data = torch.from_numpy(np.fromstring(line_data, dtype=float, sep=' '))
                '''
                x_min = min(x_min, data[0::3].min().item())
                y_min = min(y_min, data[1::3].min().item())
                z_min = min(z_min, data[2::3].min().item())
                x_max = max(x_max, data[0::3].max().item())
                y_max = max(y_max, data[1::3].max().item())
                z_max = max(z_max, data[2::3].max().item())
                '''
                data = np.deg2rad(data)
                #data_rotation = data
                data_rotation = torch.asarray([], dtype=torch.float32)
                for k, eular in self.beat_target_list.items():
                    data_rotation = torch.cat((data_rotation, data[self.beat_ori_list[k][1]-eular:self.beat_ori_list[k][1]]))
                data_all.append(data_rotation)
        #print(f"x {x_min}~{x_max} y {y_min}~{y_max} z {z_min}~{z_max}")
        data_all = torch.stack(data_all, dim=0).unsqueeze(0).float().cuda()
        return data_all #return shape: [batch_size(1), n, joints * joint_dim]
    def denormalize_pose_gpu(self, pose):
        return pose * self.std_pose_gpu + self.mean_pose_gpu
    def test_recording(self, epoch, metrics):
        if self.rank == 0: 
            pstr_curr = "Curr info >>>>  "
            pstr_best = "Best info >>>>  "

            for name, metric in metrics.items():
                if "val" in name:
                    if metric.count > 0:
                        pstr_curr += "{}: {:.3f}     \t".format(metric.name, metric.avg)
                        wandb.log({metric.name: metric.avg}, step=epoch*self.train_length)
                        if metric.avg < self.best_epochs[metric.name][0]:
                            self.best_epochs[metric.name][0] = metric.avg
                            self.best_epochs[metric.name][1] = epoch
                            other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{metric.name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
                        metric.reset()
            for k, v in self.best_epochs.items():
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, v[0], v[1])
            logger.info(pstr_curr)
            logger.info(pstr_best)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        sum_inference_frames = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        align = 0 
        if not self.read_from_out_file:
            self.model.eval()
            ffModel = other_tools.FrameByFrameModel(self.args, self.model)
        self.metric_calculator.reset()
        logger.info("Test score for \"" + self.test_model.name + "\"")
        sum_inference_time = 0
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, batch_data in enumerate(self.test_loader):
                tar_pose = batch_data["pose"].cuda()
                batch_size = tar_pose.shape[0]
                in_head = batch_data["head"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
                in_sem = batch_data["sem"].cuda() # if self.sem_rep is not None else None
                file_id = batch_data["id_pose"][0]
                
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                in_audio = in_audio.reshape(1, -1)
                
                
                if self.read_from_out_file:
                    out_file_path = self.out_file_folder_path + self.out_file_id_prefix + file_id + self.out_file_id_suffix
                    out_pose = self.readBVH(out_file_path)
                    tmpshape = out_pose.shape
                    out_pose = rotation_tools.matrix_to_rotation_6d(rotation_tools.euler_angles_to_matrix(out_pose.reshape([-1, 3]), "XYZ"))
                    out_pose = out_pose.reshape(tmpshape[0], tmpshape[1], self.train_pose_dims)
                else:
                    ffModel.reset(batch_size)
                    start_time = time.time()
                    ffModel.inputMultiFrame(tar_pose, in_audio, in_facial, in_id, in_word, in_emo, in_head)
                    sum_inference_time += time.time() - start_time
                    out_pose = self.denormalize_pose_gpu(ffModel.out_pose_vec)
                tar_pose = self.denormalize_pose_gpu(tar_pose)
                
                print(f"id {file_id} out shape {out_pose.shape} tar shape {tar_pose.shape}")
                # frame length alignment
                if self.remove_tar_first_frame:
                    tar_pose = tar_pose[:, 1:, :]
                if self.test_model == TestModel.DIFFSHEG:
                    out_pose = out_pose[:, :tar_pose.shape[1], :]
                # head bones skip for IORGesture(FRGG)
                if self.skip_head_bones: # only works on spine_neck_141 skeleton
                    out_pose[:, :, 0:(self.args.train_joint_dims * 3)] = tar_pose[:, :, 0:(self.args.train_joint_dims * 3)]
                
                # repeat conversion test tar
                #original_tar_pose = tar_pose
                #tar_pose = rotation_tools.rotation_6d_to_matrix(tar_pose.reshape([-1, 6]))
                #tar_pose = rotation_tools.matrix_to_rotation_6d(tar_pose).reshape(original_tar_pose.shape)
                #original_tar_pose = tar_pose
                #tar_pose = rotation_tools.rotation_6d_to_matrix(tar_pose.reshape([-1, 6]))
                #tar_pose = rotation_tools.matrix_to_rotation_6d(tar_pose).reshape(original_tar_pose.shape)
                # repeat conversion test out
                original_out_pose = out_pose
                out_pose = rotation_tools.rotation_6d_to_matrix(out_pose.reshape([-1, 6]))
                out_pose = rotation_tools.matrix_to_rotation_6d(out_pose).reshape(original_out_pose.shape)
                
                diff = (original_out_pose - out_pose).abs()
                #print(f"out pose rot6d repeat conversion diff {diff.mean()} max {diff.max()}")
                '''
                original_out_pose = out_pose
                out_pose = rotation_tools.rotation_6d_to_matrix(out_pose.reshape([-1, 6]))
                out_pose = rotation_tools.matrix_to_rotation_6d(out_pose).reshape(original_out_pose.shape)
                
                diff = (original_out_pose - out_pose).abs()
                print(f"2nd conversion diff {diff.mean()} max {diff.max()}")
                '''
                out_pose = out_pose.reshape([1, -1, self.train_pose_dims]) # shape[0] == 1 required for slicing
                tar_pose = tar_pose.reshape([1, -1, self.train_pose_dims])
                self.metric_calculator.add(tar_pose, out_pose, in_audio, in_sem)
                sum_inference_frames += out_pose.shape[1]

                #break

        align = self.metric_calculator.beatalign()
        logger.info(f"align score: {align}")
        srgr = self.metric_calculator.srgr()
        logger.info(f"srgr score: {srgr}")
        l1div = self.metric_calculator.l1div()
        logger.info(f"l1div score: {l1div}")
        fid = self.metric_calculator.fid()
        logger.info(f"fid score: {fid}")
        end_time = time.time() - start_time
        logger.info(f"total run time: {int(end_time)} s total inference time: {int(sum_inference_time)} for {int(sum_inference_frames/self.pose_fps)} s motion")

@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args) 
    
    trainer = BaseTrainer(args) 
    logger.info("Testing from ckpt ...")
    epoch = 9999
    trainer.test(epoch)
              
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]=str(other_tools.find_free_port(2222))
    args = config.parse_args()
    main_worker(0, 1, args)