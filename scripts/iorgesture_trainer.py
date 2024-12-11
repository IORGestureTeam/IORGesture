import train
import os
import time
import math
import csv
import sys
import warnings
import random
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import pprint
from loguru import logger

from utils import config, logger_tools, rotation_tools, other_tools, metric
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        self.alignmenter = metric.alignment(args, 0.3, 2)
        self.stride = args.stride
        self.pose_length = args.pose_length

        # 假设的骨骼长度（这些值应该根据真实人体测量数据进行调整）
        self.SHOULDER_TO_ELBOW_LENGTH = 0.3  # 单位：米
        self.ELBOW_TO_WRIST_LENGTH = 0.25  # 单位：米
        
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'),
            'reg': other_tools.AverageMeter('reg'),
            'kld': other_tools.AverageMeter('kld'),
        }
    
    def euler_ang_diff(self, pose1, pose2):
        pose_diff = pose1 - pose2
        pose_diff = (pose_diff + 180) % 360 - 180 # assume max angle vel per frame < 180
        return pose_diff
    def vel_euler_ang(self, pose):
        vel = self.euler_ang_diff(pose[:, 1:, :], pose[:, :-1, :])
        return vel
    def direct_vel(self, pose):
        vel = pose[:, 1:, :] - pose[:, :-1, :]
        return vel
    
    def arm_sym_diff_cossin(self, pose):
        arm1 = pose[..., 25*6:25*6+3*6]
        arm2 = pose[..., 3*6:3*6+3*6].clone()
        arm2[..., 25*6+3:25*6+3*6:6] *= -1
        arm2[..., 25*6+5:25*6+3*6:6] *= -1
        return arm1 - arm2
    def arm_sym_diff_euler(self, pose):
        arm1 = pose[..., 25*3:25*3+3*3]
        arm2 = pose[..., 3*3:3*3+3*3].clone()
        arm2[..., 25*3+1:25*3+3*3:3] *= -1
        arm2[..., 25*3+2:25*3+3*3:3] *= -1
        return self.euler_ang_diff(arm1, arm2)
    
    def train(self, epoch):
        use_adv = bool(epoch>=self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        ffModel = other_tools.FrameByFrameModel(self.args, self.model)
        #l2loss = torch.nn.MSELoss()
        for its, batch_data in enumerate(self.train_loader):
#             if its+1 == its_len and tadr_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
            in_id = batch_data["id"].cuda() if self.speaker_id else None
            tar_pose = batch_data["pose"].cuda()
            batch_size = tar_pose.shape[0]
            logger.info(f"{its}/{batch_size}")
            tar_pose_wo_pre = tar_pose[:, self.pre_frames:, :]
            # head rot (based on spine_neck_141)
            in_head = batch_data["head"].cuda()
            in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
            in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
            in_word = batch_data["word"].cuda() if self.word_rep is not None else None
            in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
            #in_sem = batch_data["sem"][:, self.pre_frames:].cuda() if self.sem_rep is not None else None

            t_data = time.time() - t_start 
            
            # --------------------------- d training --------------------------------- #
            d_loss_final = 0
            if use_adv:
                self.opt_d.zero_grad()
                ffModel.reset(batch_size)
                ffModel.inputMultiFrame(tar_pose, in_audio, in_facial, in_id, in_word, in_emo, in_head)
                out_pose = ffModel.out_pose_vec
                out_pose_wo_pre = out_pose[:, self.pre_frames:, :]
                out_d_fake = self.d_model(out_pose_wo_pre)
                # d_fake_for_d = self.adv_loss(out_d_fake, fake_gt)
                out_d_real = self.d_model(tar_pose_wo_pre)
                # d_real_for_d = self.adv_loss(out_d_real, real_gt)
                d_loss_adv = torch.sum(-torch.mean(torch.log(out_d_real + 1e-8) + torch.log(1 - out_d_fake + 1e-8)))
                d_loss_final += d_loss_adv
                self.loss_meters['dis'].update(d_loss_final.item()) # we ignore batch_size here
                d_loss_final.backward()
                self.opt_d.step()
                # if lrs_d is not None: lrs_d.step()

            # --------------------------- g training --------------------------------- #
            self.opt.zero_grad()
            g_loss_final = 0
            ffModel.reset(batch_size)
            ffModel.inputMultiFrame(tar_pose, in_audio, in_facial, in_id, in_word, in_emo, in_head)
            out_pose = ffModel.out_pose_vec
            out_pose_wo_pre = out_pose[:, self.pre_frames:, :]

            #tar_euler_wo_pre = other_tools.cos_sin_to_euler(self.denormalize_pose_gpu(tar_pose_wo_pre))
            #out_euler_wo_pre = other_tools.cos_sin_to_euler(self.denormalize_pose_gpu(out_pose_wo_pre))
            #if self.sem_rep is not None:
            #    huber_value = self.rec_loss(tar_pose_wo_pre*(in_sem.unsqueeze(2)+1), out_pose_wo_pre*(in_sem.unsqueeze(2)+1))
            #else: huber_value = self.rec_loss(tar_pose_wo_pre, out_pose_wo_pre)
            # full pose angle loss
            #huber_value_old = self.rec_loss(tar_pose_wo_pre, out_pose_wo_pre) * self.rec_weight
            tar_pose_wo_pre_denorm = self.denormalize_pose_gpu(tar_pose_wo_pre)
            out_pose_wo_pre_denorm = self.denormalize_pose_gpu(out_pose_wo_pre)
            
            pose_rec_loss = self.rec_loss(tar_pose_wo_pre, out_pose_wo_pre) * self.rec_weight

            #pose_diff = self.sin_cos_loss(tar_pose_wo_pre, out_pose_wo_pre)
            #huber_value = self.rec_loss(pose_diff, torch.zeros_like(pose_diff)) * self.rec_weight
            #huber_value = self.angle_diff(tar_pose_wo_pre, out_pose_wo_pre) * self.rec_weight
            self.loss_meters['rec'].update(pose_rec_loss.item())
            g_loss_final += pose_rec_loss
            
            # full pose vel / acc loss
            # must use euler angles making vel / acc loss
            tar_bone_vel = self.direct_vel(tar_pose_wo_pre)
            out_bone_vel = self.direct_vel(out_pose_wo_pre)
            #t1 = self.direct_vel(tar_pose_wo_pre)
            #t2 = self.direct_vel(out_pose_wo_pre)
            tar_bone_acc = self.direct_vel(tar_bone_vel)
            out_bone_acc = self.direct_vel(out_bone_vel)
            #pose_vel_loss = l2loss(out_bone_vel, tar_bone_vel) * 1000
            pose_vel_loss = self.rec_loss(out_bone_vel, tar_bone_vel) * 1000
            #pose_direct_vel_loss = (t1 - t2).abs().mean() * 1000
            pose_acc_loss = self.rec_loss(out_bone_acc, tar_bone_acc) * 1000
            #g_loss_final += pose_vel_loss # tmp removed vel loss
            g_loss_final += pose_vel_loss
            g_loss_final += pose_acc_loss
            
            # arm sym diff loss
            #tar_arm_diff = self.arm_sym_diff_cossin(tar_pose_wo_pre)
            #out_arm_diff = self.arm_sym_diff_cossin(out_pose_wo_pre)
            #tar_arm_diff = self.arm_sym_diff_euler(tar_pose_euler_wo_pre)
            #out_arm_diff = self.arm_sym_diff_euler(out_pose_euler_wo_pre)
            #logger.info(f"sym diff euler ang mean out {out_arm_diff.abs().mean()} tar {tar_arm_diff.abs().mean()}")
            #arm_sym_diff_loss = (tar_arm_diff - out_arm_diff).abs().mean()
            #g_loss_final += arm_sym_diff_loss
            '''
            # arm pose vel loss
            tar_bone_vel = self.arm_vel(tar_pose_wo_pre)
            out_bone_vel = self.arm_vel(out_pose_wo_pre)
            arm_vel_loss = (tar_bone_vel - out_bone_vel).abs().mean() * 2500
            g_loss_final += arm_vel_loss
            '''
            #logger.info(f"huber {huber_value} vel loss {pose_vel_loss} arm sub loss {arm_sub_loss} (original: {abs_arm_sub_diff.mean() * 100})")
            #logger.info(f"tar max vel {tar_bone_vel.max()} min {tar_bone_vel.min()} out max vel {out_bone_vel.max()} min {out_bone_vel.min()} ")
            logger.info(f"rec {pose_rec_loss} vel {pose_vel_loss} acc {pose_acc_loss} sym X)")

            if use_adv:
                dis_out = self.d_model(out_pose_wo_pre)
                d_fake_value = -torch.mean(torch.log(dis_out + 1e-8)) # self.adv_loss(out_d_fake, real_gt) # here 1 is real
                d_fake_value *= self.adv_weight * d_fake_value
                self.loss_meters['gen'].update(d_fake_value.item())
                g_loss_final += d_fake_value
                
#                 latent_out = self.eval_model(out_pose)
#                 latent_ori = self.eval_model(tar_pose)
#                 huber_fid_loss = self.rec_loss(latent_out, latent_ori) * self.fid_weight
#                 self.loss_meters[4].update(huber_fid_loss.item())
#                 g_loss_final += huber_fid_loss
            
            self.loss_meters['all'].update(g_loss_final.item())
            g_loss_final.backward()
            if self.grad_norm != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.opt.step()
            # if lrs is not None: lrs.step() 
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            lr_d = self.opt_d.param_groups[0]['lr']
            
            # --------------------------- recording ---------------------------------- #
            if its % self.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, lr_d, t_data, t_train, mem_cost)
            #if its == 1:break
            
            #print(f"edit check {(original_tar - tar_pose_wo_pre).abs().sum()}")
        self.opt_s.step(epoch)
        self.opt_d_s.step(epoch)
    
    
    def val(self, epoch):
        self.model.eval()
        ffModel = other_tools.FrameByFrameModel(self.args, self.model)
        self.metric_calculator.reset()
        with torch.no_grad():
            its_len = len(self.val_loader)
            
            for its, batch_data in enumerate(self.val_loader):
#                 if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
                tar_pose = batch_data["pose"].cuda()
                tar_pose_wo_pre = tar_pose[:, self.pre_frames:, :]
                in_head = batch_data["head"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
                in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None
                batch_size = tar_pose.shape[0]

                ffModel.reset(batch_size)
                ffModel.inputMultiFrame(tar_pose, in_audio, in_facial, in_id, in_word, in_emo, in_head)
                out_pose = ffModel.out_pose_vec
                out_pose_wo_pre = out_pose[:, self.pre_frames:, :]
                #print(f"out_pose val shape {out_pose_wo_pre.shape}") # 256 18 282
                out_pose_wo_pre = self.denormalize_pose_gpu(out_pose_wo_pre)
                tar_pose_wo_pre = self.denormalize_pose_gpu(tar_pose_wo_pre)
                '''
                out_pose_wo_pre = rotation_tools.rotation_6d_to_matrix(self.denormalize_pose_gpu(out_pose_wo_pre).reshape([-1, 6]))
                out_pose_wo_pre = rotation_tools.matrix_to_euler_angles(out_pose_wo_pre, "XYZ") * 180 / torch.pi
                tar_pose_wo_pre = rotation_tools.rotation_6d_to_matrix(self.denormalize_pose_gpu(tar_pose_wo_pre).reshape([-1, 6]))
                tar_pose_wo_pre = rotation_tools.matrix_to_euler_angles(tar_pose_wo_pre, "XYZ") * 180 / torch.pi
                '''
                #print(f"<val>A out shape {out_pose_wo_pre.shape} ")
                out_pose_wo_pre = out_pose_wo_pre.reshape([1, -1, self.train_pose_dims]) # shape[0] == 1 required for slicing
                tar_pose_wo_pre = tar_pose_wo_pre.reshape([1, -1, self.train_pose_dims])
                
                #print(f"<val>B out shape {out_pose_wo_pre.shape} ")
                self.metric_calculator.add(tar_pose_wo_pre, out_pose_wo_pre, in_audio, in_sem) #TODO: in_audio trim
                huber_value = self.rec_loss(tar_pose_wo_pre, out_pose_wo_pre)
                huber_value *= self.rec_weight
                self.loss_meters['rec_val'].update(huber_value.item())
                #if its == 1:break
            fid = self.metric_calculator.fid()
            self.loss_meters['fid_val'].update(fid)
            self.val_recording(epoch, self.loss_meters)
                
        
    def test(self, epoch, quicklook=False): # usages not found
        loss_meters_single_test = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'),
            'reg': other_tools.AverageMeter('reg'),
            'kld': other_tools.AverageMeter('kld'),
        } 
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        t_start = 10
        t_end = 500
        align = 0 
        self.model.eval()
        ffModel = other_tools.FrameByFrameModel(self.args, self.model)
        self.metric_calculator.reset()
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, batch_data in enumerate(self.test_loader):
                id_pose = batch_data["id_pose"][0] # batch_size=1
                tar_pose = batch_data["pose"].cuda()
                in_head = batch_data["head"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
                in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None
                batch_size = tar_pose.shape[0]
                
                in_audio = in_audio.reshape(1, -1)

                ffModel.reset(batch_size)
                ffModel.inputMultiFrame(tar_pose, in_audio, in_facial, in_id, in_word, in_emo, in_head)
                out_pose = ffModel.out_pose_vec
                #out_pose = tar_pose
                
                # write file

                #logger.info(f"out_dir_vec shape {out_dir_vec.shape}")
                
                out_pose = self.denormalize_pose_gpu(out_pose)
                tar_pose = self.denormalize_pose_gpu(tar_pose)
                
                out_pose = out_pose.reshape([1, -1, self.train_pose_dims]) # shape[0] == 1 required for slicing
                tar_pose = tar_pose.reshape([1, -1, self.train_pose_dims])
                self.metric_calculator.add(tar_pose, out_pose, in_audio, in_sem) #TODO: in_audio trim
                #logger.info(f"out_euler shape {out_euler.shape}")
                out_pose = rotation_tools.rotation_6d_to_matrix(out_pose.reshape([-1, 6]))
                out_pose = rotation_tools.matrix_to_euler_angles(out_pose, "XYZ") * 180 / torch.pi
                out_pose = out_pose.reshape(-1, self.output_pose_dims).cpu().numpy()
                #out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                #print(f"std {self.std_pose[15]} mean {self.mean_pose[15]} raw {tar_pose.reshape(-1)[max_id]} dn {self.denormalize_pose_gpu(tar_pose).reshape(-1)[max_id]}")
                #print(f"out final {out_final.reshape(-1)[max_id]}")
                
                total_length += out_pose.shape[0]
                
                #onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio.cpu().numpy().reshape(-1), t_start, t_end, True)
                #beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(out_euler, t_start, t_end, self.pose_fps, True)
                #align += self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.pose_fps)
                with open(f"{results_save_path}result_raw_{id_pose}.bvh", 'w+') as f_real:
                    for line_id in range(out_pose.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        #logger.info(f"line: {line_data}")
                        f_real.write(line_data[1:-2]+'\n')
                if quicklook:
                    break
        #align_avg = align/len(self.test_loader)
        logger.info(f"align score: {self.metric_calculator.beatalign()}")

        fid = self.metric_calculator.fid()
        loss_meters_single_test['fid_val'].update(fid)
        self.val_recording(epoch, loss_meters_single_test)

        data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")
               
    @staticmethod
    def diversity(output, clips):
        pass
    
    @staticmethod
    def SRGR(output, target, weight, alpha=0.2):
        pass