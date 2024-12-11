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
import wandb

from utils import config, logger_tools, other_tools, metric, rotation_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.suppress_file_writes = False
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
        #self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        #self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
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
        
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        self.model = getattr(model_module, args.g_name)(args)
        other_tools.load_checkpoints(self.model, args.root_path+args.test_ckpt, args.g_name)
        self.model = torch.nn.DataParallel(self.model, args.gpus).cuda()
        if self.rank == 0:
            logger.info(self.model)
            wandb.watch(self.model)
            logger.info(f"init {args.g_name} success")
        
        self.metric_calculator = other_tools.Metric_calculator(args, self.rank)
    
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
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        align = 0 
        self.model.eval()
        ffModel = other_tools.FrameByFrameModel(self.args, self.model)
        self.metric_calculator.reset()
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
                
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                in_audio = in_audio.reshape(1, -1)
                
                ffModel.reset(batch_size)
                ffModel.inputMultiFrame(tar_pose, in_audio, in_facial, in_id, in_word, in_emo, in_head)
                #smooth test
                out_pose = ffModel.out_pose_vec
                out_pose = self.denormalize_pose_gpu(out_pose)
                tar_pose = self.denormalize_pose_gpu(tar_pose)
                
                out_pose = out_pose.reshape([1, -1, self.train_pose_dims]) # shape[0] == 1 required for slicing
                tar_pose = tar_pose.reshape([1, -1, self.train_pose_dims])
                self.metric_calculator.add(tar_pose, out_pose, in_audio, in_sem)
                #logger.info(f"out_euler shape {out_euler.shape}")
                out_pose = rotation_tools.rotation_6d_to_matrix(out_pose.reshape([-1, 6]))
                out_pose = rotation_tools.matrix_to_euler_angles(out_pose, "XYZ") * 180 / torch.pi
                out_pose = out_pose.reshape(-1, self.output_joint_dims).cpu().numpy()
                '''
                if self.trainer_name == "multi":
                    out_dir_vec, *_ = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_word=in_word, in_id=in_id))
                else:
                    out_dir_vec = self.model(**dict(pre_seq=pre_pose, in_audio=in_audio, in_text=in_word, in_facial=in_facial, in_id=in_id, in_emo=in_emo))
                '''
                #print(f"<test> out_pose shape {out_pose.shape}")
                #print(f"arm1 ea {out_pose_eular_angles[0, :64, 15:18]}")
                #print(f"arm1 quat {self.denormalize_pose_gpu(out_pose)[0, :64, 20:24]}")
                
                total_length += out_pose.shape[0]
                
                if not self.suppress_file_writes:
                    with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                        for line_id in range(out_pose.shape[0]): #,args.pre_frames, args.pose_length
                            line_data = np.array2string(out_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                            f_real.write(line_data[1:-2]+'\n')
                break
        align = self.metric_calculator.beatalign()
        logger.info(f"align score: {align}")
        srgr = self.metric_calculator.srgr()
        logger.info(f"srgr score: {srgr}")
        l1div = self.metric_calculator.l1div()
        logger.info(f"l1div score: {l1div}")
        fid = self.metric_calculator.fid()
        logger.info(f"fid score: {fid}")
        if not self.suppress_file_writes:
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")

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
    os.environ["MASTER_PORT"]='2222'
    args = config.parse_args()
    main_worker(0, 1, args)