import os
import numpy as np
import math
import random
import torch
import csv
import pprint
from loguru import logger
import wandb
from utils import metric, rotation_tools
from collections import OrderedDict
from dataloaders import data_tools
import socket

def smooth_SMA(pose):
    # Requires [<batch_size>, <pose_length>, *] shape
    window_size = 3
    stored_pose = torch.zeros([pose.shape[0], window_size, pose.shape[2]], dtype=pose.dtype, device=pose.device)
    smoothed_pose = torch.zeros_like(pose)
    for frame_ite in range(pose.shape[1]):
        stored_pose[:, 1:, :] = stored_pose[:, :-1, :].clone()
        stored_pose[:, 0, :] = pose[:, frame_ite, :]
        avg = stored_pose[:, -(frame_ite + 1):, :].mean(dim=1)
        smoothed_pose[:, frame_ite, :] = avg
    return smoothed_pose

class Metric_calculator:
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.stride = args.stride
        self.joint_nums = args.joint_nums
        self.eval_rotation = args.eval_rotation
        self.eval_joint_dims = args.eval_joint_dims
        self.eval_pose_dims = args.joint_nums * args.eval_joint_dims
        self.pose_fps = args.pose_fps
        # pose_length and mean/std used in AE training are differ from the main model
        self.ae_pose_length = args.ae_pose_length
        ae_mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_eval_rep}/{args.pose_eval_rep}_mean.npy")
        ae_std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_eval_rep}/{args.pose_eval_rep}_std.npy")
        self.ae_mean_pose_gpu = torch.tensor(ae_mean_pose).cuda().float()
        self.ae_std_pose_gpu = torch.tensor(ae_std_pose).cuda().float()
        self.t_start = 10
        self.t_end = 500
    def reset(self):
        self.sample_count = 0
        self.latent_out_all = self.latent_ori_all = None
        if self.args.e_name is not None:
            eval_model_module = __import__(f"models.{self.args.eval_model}", fromlist=["something"])
            self.eval_model = getattr(eval_model_module, self.args.e_name)(self.args).cuda()
            if self.rank == 0:
                load_checkpoints(self.eval_model, self.args.root_path+self.args.e_path, self.args.e_name)
            #self.eval_model = torch.nn.DataParallel(self.eval_model, self.args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.eval_model)
                wandb.watch(self.eval_model)
                logger.info(f"init {self.args.e_name} success")
            self.eval_model.eval()
        self.align = 0
        self.alignmenter = metric.alignment(self.args, 0.3, 2)
        self.srgr_calculator = metric.SRGR(self.args.srgr_threshold, self.joint_nums, self.eval_joint_dims)
        self.l1_calculator = metric.L1div()
    def ae_normalize_pose_gpu(self, pose):
        return (pose - self.ae_mean_pose_gpu) / self.ae_std_pose_gpu
    def add(self, tar_pose, out_pose, in_audio, in_sem):
        # Requires [<batch_size>, <pose_length>, pose_dims] shape denormalized rot6d pose
        self.sample_count += 1
        # BeatAlign
        shape = [tar_pose.shape[0], tar_pose.shape[1], self.eval_pose_dims]
        # If using axis angle to eval
        #tar_pose = rotation_tools.matrix_to_axis_angle(rotation_tools.euler_angles_to_matrix(tar_pose.reshape([-1, 3]), 'XYZ'))
        #out_pose = rotation_tools.matrix_to_axis_angle(rotation_tools.euler_angles_to_matrix(out_pose.reshape([-1, 3]), 'XYZ'))
        #if self.eval_rotation == rotation_tools.RotationType.AXIS_ANGLE.name:
        #    tar_pose = rotation_tools.matrix_to_axis_angle(rotation_tools.rotation_6d_to_matrix(tar_pose.reshape([-1, 6]))).reshape(shape)
        #    out_pose = rotation_tools.matrix_to_axis_angle(rotation_tools.rotation_6d_to_matrix(out_pose.reshape([-1, 6]))).reshape(shape)
        np_cat_out = out_pose.cpu().numpy().reshape(-1, self.eval_pose_dims)
        np_cat_tar = tar_pose.cpu().numpy().reshape(-1, self.eval_pose_dims)
        onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio.cpu().numpy().reshape(-1), self.t_start, self.t_end, True)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(np_cat_out, self.t_start, self.t_end, self.pose_fps, True)
        self.align += self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.pose_fps)
        # SRGR
        _ = self.srgr_calculator.run(np_cat_out, np_cat_tar, in_sem.cpu().numpy())
        # L1DIV
        cat_results = slice_pose(self.stride, self.ae_pose_length, out_pose)
        cat_targets = slice_pose(self.stride, self.ae_pose_length, tar_pose)
        _ = self.l1_calculator.run(cat_results.cpu().numpy().reshape(-1, self.eval_pose_dims))
        # FID
        #latent_out = self.eval_model(self.ae_normalize_pose_gpu(cat_results)).cpu().numpy()
        #latent_ori = self.eval_model(self.ae_normalize_pose_gpu(cat_targets)).cpu().numpy()
        latent_out = self.eval_model.encoder(self.ae_normalize_pose_gpu(cat_results)).reshape(-1, self.args.vae_length).cpu().numpy()
        latent_ori = self.eval_model.encoder(self.ae_normalize_pose_gpu(cat_targets)).reshape(-1, self.args.vae_length).cpu().numpy()
        if self.latent_out_all is None:
            self.latent_out_all = latent_out
            self.latent_ori_all = latent_ori
        else:
            self.latent_out_all = np.concatenate([self.latent_out_all, latent_out], axis=0)
            self.latent_ori_all = np.concatenate([self.latent_ori_all, latent_ori], axis=0)
    def beatalign(self):
        align_avg = self.align / self.sample_count
        return align_avg
    def fid(self):
        fid = data_tools.FIDCalculator.frechet_distance(self.latent_out_all, self.latent_ori_all)
        return fid
    def srgr(self):
        srgr = self.srgr_calculator.avg()
        return srgr
    def l1div(self):
        l1div = self.l1_calculator.avg()
        return l1div
def slice_pose(stride, pose_length, pose):
    # Receive [N, M, 141] euler angles pose
    #print(f"pose shape {pose.shape} pose_length {pose_length} stride {stride}")
    num_divs = (pose.shape[1] - pose_length) // stride + 1
    for i in range(num_divs):
        pose_slice = pose[:,i*stride:i*stride+pose_length, :]
        if i == 0:
            cat_pose = pose_slice
        else:
            cat_pose = torch.cat([cat_pose, pose_slice], 0)
    return cat_pose

class FrameByFrameModel():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.pre_frames_plus1 = args.pre_frames + 1
        self.audio_length_per_pose_frame = math.floor(args.audio_fps / args.pose_fps)
        self.pose_train_dims = args.joint_nums * args.train_joint_dims
        self.facial_dims = args.facial_dims
    
    def reset(self, batch_size):
        self.batch_size = batch_size
        self.frame_count = 0
        self.out_pose_vec = None
        self.in_pre_pose = torch.zeros([self.batch_size, 0, self.pose_train_dims + 1]).cuda()
        self.in_head = torch.zeros([self.batch_size, 0, 6]).cuda()
        self.in_facial = torch.zeros([self.batch_size, 0, self.facial_dims]).cuda() if self.args.facial_rep is not None else None
        self.in_audio = torch.zeros([self.batch_size, 0]).cuda() if self.args.audio_rep is not None else None
        self.in_word = torch.zeros([self.batch_size, 0], dtype=torch.int32).cuda() if self.args.word_rep is not None else None
        self.in_emo = torch.zeros([self.batch_size, 0], dtype=torch.int32).cuda() if self.args.emo_rep is not None else None
        self.lstm_hidden = self.getInitHidden()
        self.lstm_hands_hidden = self.getInitHidden()
    def getInitHidden(self):
        return (torch.zeros(self.args.n_layer, self.batch_size, self.args.hidden_size).cuda(),
                  torch.zeros(self.args.n_layer, self.batch_size, self.args.hidden_size).cuda())
    def input1Frame(self, tar_pose_1frame, in_audio_1frame, in_facial_1frame, in_id, in_word_1frame, in_emo_1frame, in_head_rot_1frame):
        self.frame_count += 1
        # Add new row
        self.in_pre_pose = torch.cat((self.in_pre_pose, torch.zeros(self.batch_size, 1, self.pose_train_dims + 1).cuda()), 1)
        self.in_head = torch.cat((self.in_head, torch.zeros(self.batch_size, 1, self.in_head.shape[-1]).cuda()), 1)
        self.in_facial = torch.cat((self.in_facial, torch.zeros(self.batch_size, 1, self.facial_dims).cuda()), 1)
        if self.in_word is not None:
            self.in_word = torch.cat((self.in_word, torch.zeros([self.batch_size, 1], dtype=torch.int32).cuda()), 1)
        self.in_emo = torch.cat((self.in_emo, torch.zeros([self.batch_size, 1], dtype=torch.int32).cuda()), 1)
        # Put
        self.in_head[:, -1:, :] = in_head_rot_1frame
        self.in_audio = torch.cat((self.in_audio, in_audio_1frame), 1)
        self.in_facial[:, -1:, :] = in_facial_1frame
        if self.in_word is not None:
            self.in_word[:, -1:] = in_word_1frame
        self.in_emo[:, -1:] = in_emo_1frame
        # Wait for accumurate enough frames to begin prediction
        if self.frame_count <= self.args.pre_frames:
            self.in_pre_pose[:, -1:, :-1] = tar_pose_1frame
            self.in_pre_pose[:, -1:, -1] = 1
            new_frame_out_pose_vec = tar_pose_1frame[:, 0, :]
        # prediction
        else:
            #print(f"in_audio shape {self.in_audio[:, -self.audio_length_per_pose_frame*self.pre_frames_plus1:].shape}, split {-self.audio_length_per_pose_frame*self.pre_frames_plus1}")
            out_dir_vec, self.lstm_hidden, self.lstm_hands_hidden = self.model(**dict(
                pre_seq=self.in_pre_pose[:, -self.pre_frames_plus1:, :],
                in_audio=self.in_audio[:, -self.audio_length_per_pose_frame*self.pre_frames_plus1:],
                in_text=self.in_word[:, -self.pre_frames_plus1:] if self.in_word is not None else None,
                in_facial=self.in_facial[:, -self.pre_frames_plus1:, :],
                in_id=in_id,
                in_emo=self.in_emo[:, -self.pre_frames_plus1:]),
                in_head=self.in_head[:, -self.pre_frames_plus1:, :],
                lstm_hidden=self.lstm_hidden,
                lstm_hands_hidden=self.lstm_hands_hidden)
            new_frame_out_pose_vec = out_dir_vec[:, :]
            # record current result for next
            #logger.info(f"A shape {self.in_pre_pose[:, -1, :-1].shape} B shape {new_frame_out_pose_vec.shape}")
            self.in_pre_pose[:, -1, :-1] = new_frame_out_pose_vec
            self.in_pre_pose[:, -1, -1] = 1
        if self.out_pose_vec is None:
            self.out_pose_vec = new_frame_out_pose_vec.unsqueeze(1)
        else:
            self.out_pose_vec = torch.cat((self.out_pose_vec, new_frame_out_pose_vec.unsqueeze(1)), axis=1)
            
    def inputMultiFrame(self, tar_pose, in_audio, in_facial, in_id, in_word, in_emo, head_rot):
        #print(f"<FrameByFrameModel inputMultiFrame>tar_pose shape: {tar_pose.shape}")
        
        for frameItx in range(tar_pose.shape[1]):
            self.input1Frame(
                tar_pose[:, frameItx:(frameItx+1), :],
                in_audio[:, frameItx*self.audio_length_per_pose_frame:(frameItx + 1)*self.audio_length_per_pose_frame],
                in_facial[:, frameItx:(frameItx+1), :],
                in_id[:, :],
                in_word[:, frameItx:(frameItx+1)] if in_word is not None else None,
                in_emo[:, frameItx:(frameItx+1)],
                head_rot[:, frameItx:(frameItx+1), :])
            #print(f"<FrameByFrameModel inputMultiFrame>out_pose_vec shape: {self.out_pose_vec.shape} in {frameItx} / {tar_pose.shape}")

def cos_sin_to_euler(cos_sin):
    cos_tensor = cos_sin[..., ::2]  # 取所有偶数位（cz, cy, cx）
    sin_tensor = cos_sin[..., 1::2]  # 取所有奇数位（sz, sy, sx）
    z_angles = torch.atan2(sin_tensor[..., 0::3], cos_tensor[..., 0::3])
    y_angles = torch.atan2(sin_tensor[..., 1::3], cos_tensor[..., 1::3])
    x_angles = torch.atan2(sin_tensor[..., 2::3], cos_tensor[..., 2::3])
    euler_angles = torch.stack((z_angles, y_angles, x_angles), dim=-1)
    shape = euler_angles.shape
    shape = shape[:-2] + (shape[-2] * shape[-1], )
    euler_angles = euler_angles.reshape(shape)
    euler_angles = euler_angles / torch.pi * 180
    return euler_angles

def ignore_none_cat(tensors, dim=0):
    tensors = [tensor for tensor in tensors if tensor is not None]
    if not tensors:
        return torch.empty(0)
    return torch.cat(tensors, dim=dim)

def print_exp_info(args):
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"# ------------ {args.name} ----------- #")
    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("CUDA version: {}".format(torch.version.cuda))
    logger.info("{} GPUs".format(torch.cuda.device_count()))
    logger.info(f"Random Seed: {args.random_seed}")

def args2csv(args, get_head=False, list4print=[]):
    for k, v in args.items():
        if isinstance(args[k], dict):
            args2csv(args[k], get_head, list4print)
        else: list4print.append(k) if get_head else list4print.append(v)
    return list4print

def record_trial(args, csv_path, best_metric, best_epoch):
    metric_name = []
    metric_value = []
    metric_epoch = []
    list4print = []
    name4print = []
    for k, v in vars(args).items():
        list4print.append(v)
        name4print.append(k)
    
    for k, v in best_metric.items():
        metric_name.append(k)
        metric_value.append(v)
        metric_epoch.append(best_epoch[k])
    
    if not os.path.exists(csv_path):
        with open(csv_path, "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([*metric_name, *metric_name, *name4print])
            
    with open(csv_path, "a+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([*metric_value,*metric_epoch, *list4print])
        

def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic #args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled
    

def save_checkpoints(save_path, model, opt=None, epoch=None, lrs=None):
    if lrs is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),
                'lrs':lrs.state_dict(),}
    elif opt is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),}
    else:
        states = { 'model_state': model.state_dict(),}
    torch.save(states, save_path)


def load_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v
            flag=True
    if flag: 
        model.load_state_dict(new_weights)
    else:
        model.load_state_dict(states['model_state'])
    logger.info(f"load self-pretrained checkpoints for {load_name}")


def model_complexity(model, args):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (args.T_GLOBAL._DIM, args.TRAIN.CROP, args.TRAIN), 
        as_strings=False, print_per_layer_stat=False)
    logging.info('{:<30}  {:<8} BFlops'.format('Computational complexity: ', flops / 1e9))
    logging.info('{:<30}  {:<8} MParams'.format('Number of parameters: ', params / 1e6))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def find_free_port(start_port):
    port = start_port
    while True:
        try:
            # 尝试创建一个socket并绑定到该端口
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                # 如果绑定成功，说明端口未被占用
                return port
        except socket.error:
            # 如果绑定失败，说明端口可能被占用，尝试下一个端口
            port += 1