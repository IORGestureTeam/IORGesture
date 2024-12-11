import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils.skeleton import ResidualBlock, SkeletonResidual, residual_ratio, SkeletonConv, SkeletonPool, find_neighbor, build_edge_topology
from .utils.layer import ResBlock, init_weight
from dataloaders import data_tools

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )
    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, seq_length, dim, feature_length=32):
        super().__init__()
        self.seq_length = seq_length # = args.ae_pose_length
        self.base = feature_length
        self.net = nn.Sequential(
            ConvNormRelu(dim, self.base, batchnorm=True), #32
            ConvNormRelu(self.base, self.base*2, batchnorm=True), #30
            ConvNormRelu(self.base*2, self.base*2, True, batchnorm=True), #14
            nn.Conv1d(self.base*2, self.base, 3)
        )
        self.out_net = nn.Sequential(
            nn.Linear((seq_length//2-5)*self.base, self.base*4),
            #nn.Linear(12*self.base, self.base*4),  # for 34 frames
            nn.BatchNorm1d(self.base*4),
            nn.LeakyReLU(True),
            nn.Linear(self.base*4, self.base*2),
            nn.BatchNorm1d(self.base*2),
            nn.LeakyReLU(True),
            nn.Linear(self.base*2, self.base),
        )

        self.fc_mu = nn.Linear(self.base, self.base)
        self.fc_logvar = nn.Linear(self.base, self.base)

    def forward(self, poses, variational_encoding=None):
        # encode
        poses = poses.transpose(1, 2)
        #print(f"poses shape {poses.shape}")
        out = self.net(poses)
        #print(f"out shape A {out.shape}")
        out = out.flatten(1)
        #print(f"out shape B {out.shape}")
        out = self.out_net(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        #return z, mu, logvar
        return z


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False, feature_length=32):
        super().__init__()
        self.use_pre_poses = use_pre_poses
        self.feat_size = feature_length
        
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            self.feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 256),
            )
        #elif length == 34:
        else:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size*2),
                nn.BatchNorm1d(self.feat_size*2),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size*2, self.feat_size//8*length),
            )
        self.decoder_size = self.feat_size//8
        self.net = nn.Sequential(
            nn.ConvTranspose1d(self.decoder_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(self.feat_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(self.feat_size, self.feat_size*2, 3),
            nn.Conv1d(self.feat_size*2, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)
        #print(feat.shape)
        out = self.pre_net(feat)
        #print(out.shape)
        out = out.view(feat.shape[0], self.decoder_size, -1)
        #print(out.shape)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_frames = args.ae_pose_length # original AE: 34 new AE: args.pose_length - args.pre_frames
        pose_dim = args.joint_nums * args.eval_joint_dims
        feature_length = args.vae_length
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim, feature_length=feature_length)
        self.decoder = PoseDecoderConv(n_frames, pose_dim, feature_length=feature_length)

    def forward(self, pre_poses, poses, variational_encoding=False):
        poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
        out_poses = self.decoder(poses_feat, pre_poses)
        return poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

class HalfEmbeddingNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_frames = args.ae_pose_length
        pose_dim = args.joint_nums * args.eval_joint_dims # 47 * x
        feature_length = args.vae_length
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim, feature_length=feature_length)
        self.decoder = PoseDecoderConv(n_frames, pose_dim, feature_length=feature_length)

    def forward(self, poses):
        poses_feat, _, _ = self.pose_encoder(poses)
        return poses_feat
    
class VAEConv(nn.Module):
    def __init__(self, args):
        super(VAEConv, self).__init__()
        self.encoder = None
        #self.decoder = VQDecoderV3(args)
        #self.fc_mu = nn.Linear(args.vae_length, args.vae_length)
        #self.fc_logvar = nn.Linear(args.vae_length, args.vae_length)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        mu, logvar = None, None
        rec_pose = self.decoder(pre_latent)
        return pre_latent, mu, logvar, rec_pose
        '''
        return {
            "poses_feat":pre_latent,
            "rec_pose": rec_pose,
            "pose_mu": mu,
            "pose_logvar": logvar,
            }
        '''
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        return pre_latent
    
    #def decode(self, pre_latent):
        #rec_pose = self.decoder(pre_latent)
        #return rec_pose
class VAEConvMLP(VAEConv):
    def __init__(self, args):
        super(VAEConvMLP, self).__init__(args)
        pose_dim = args.joint_nums * args.eval_joint_dims # 47 * x
        self.encoder = PoseEncoderConv_EMAGE(args.ae_pose_length, pose_dim, feature_length=args.vae_length)
        self.decoder = PoseDecoderConv_EMAGE(args.ae_pose_length, pose_dim, feature_length=args.vae_length)
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        return pre_latent
"""
from Trimodal,
encoder:
    bs, n, c_in --conv--> bs, n/k, c_out_0 --mlp--> bs, c_out_1, only support fixed length
decoder:
    bs, c_out_1 --mlp--> bs, n/k*c_out_0 --> bs, n/k, c_out_0 --deconv--> bs, n, c_in
"""
class PoseEncoderConv_EMAGE(nn.Module):
    def __init__(self, length, dim, feature_length=32):
        super().__init__()
        self.base = feature_length
        self.seq_length = length
        self.net = nn.Sequential(
            ConvNormRelu(dim, self.base, batchnorm=True), #32
            ConvNormRelu(self.base, self.base*2, batchnorm=True), #30
            ConvNormRelu(self.base*2, self.base*2, True, batchnorm=True), #14     
            nn.Conv1d(self.base*2, self.base, 3)
        )
        self.out_net = nn.Sequential(
            nn.Linear((self.seq_length//2-5)*self.base, self.base*4),  # for 34 frames (12)
            nn.BatchNorm1d(self.base*4),
            nn.LeakyReLU(True),
            nn.Linear(self.base*4, self.base*2),
            nn.BatchNorm1d(self.base*2),
            nn.LeakyReLU(True),
            nn.Linear(self.base*2, self.base),
        )
        self.fc_mu = nn.Linear(self.base, self.base)
        self.fc_logvar = nn.Linear(self.base, self.base)

    def forward(self, poses, variational_encoding=None):
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        #return z, mu, logvar
        return z
class PoseDecoderConv_EMAGE(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False, feature_length=32):
        super().__init__()
        self.use_pre_poses = use_pre_poses
        self.feat_size = feature_length
        
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            self.feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size),
                nn.BatchNorm1d(self.feat_size),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size, self.feat_size//8*64),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size*2),
                nn.BatchNorm1d(self.feat_size*2),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size*2, self.feat_size//8*34),
            )
        elif length == 32:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size*2),
                nn.BatchNorm1d(self.feat_size*2),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size*2, self.feat_size//8*32),
            )
        else:
            assert False
        self.decoder_size = self.feat_size//8
        self.net = nn.Sequential(
            nn.ConvTranspose1d(self.decoder_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(self.feat_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(self.feat_size, self.feat_size*2, 3),
            nn.Conv1d(self.feat_size*2, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)
        #print(feat)
        out = self.pre_net(feat)
        #print(out.shape)
        out = out.view(feat.shape[0], self.decoder_size, -1)
        #print(out.shape)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out
import os
class VAESKConv(VAEConv):
    def __init__(self, args):
        super(VAESKConv, self).__init__(args)
        #smpl_fname = args.data_path_1+'smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        #smpl_data = np.load(smpl_fname, encoding='latin1')
        #parents = smpl_data['kintree_table'][0].astype(np.int32)
        #print(f"smpl_fname: {smpl_fname}")
        #print(f"parents: {parents.shape}")
        #edges = build_edge_topology(parents)
        edges = readBoneStructure("spine_neck_141_boneid_structure", args.joint_nums)
        #print(f"edges: {edges}")
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)
class LocalEncoder(nn.Module):
    def __init__(self, args, topology):
        super(LocalEncoder, self).__init__()
        args.channel_base = 6
        args.activation = "tanh"
        args.use_residual_blocks=True
        args.z_dim=1024
        args.temporal_scale=8
        args.kernel_size=4
        args.vae_layer=4
        args.num_layers=args.vae_layer
        args.skeleton_dist=2
        args.extra_conv=0
        # check how to reflect in 1d
        args.padding_mode="constant"
        args.skeleton_pool="mean"
        args.upsampling="linear"


        self.topologies = [topology]
        self.channel_base = [args.channel_base]

        self.channel_list = []
        self.edge_num = [len(topology)]
        #print(f'topology: {topology}')
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        # self.convs = []

        kernel_size = args.kernel_size
        kernel_even = False if kernel_size % 2 else True
        padding = (kernel_size - 1) // 2
        bias = True
        args.vae_grow = [1,1,2,1]
        self.grow = args.vae_grow
        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1]*self.grow[i])

        for i in range(args.num_layers):
            #print(f'LocalEncoder {i} / {args.num_layers}')
            seq = []
            #print(f'topologies[i]: {self.topologies[i]}')
            neighbour_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            #print(f'neighbour_list: {neighbour_list}')
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            #print(f"out_channels {out_channels} self.channel_base[i + 1] {self.channel_base[i + 1]} self.edge_num[i] {self.edge_num[i]}")
            #print(f"self.topologies {self.topologies}")
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == args.num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool)

            if args.use_residual_blocks:
                # (T, J, D) => (T/2, J', 2D)
                #print(f'A out_channels: {out_channels}')
                seq.append(SkeletonResidual(self.topologies[i], neighbour_list, joint_num=self.edge_num[i], in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=2, padding=padding, padding_mode=args.padding_mode, bias=bias,
                                            extra_conv=args.extra_conv, pooling_mode=args.skeleton_pool, activation=args.activation, last_pool=last_pool))
            else:
                for extra_conv_loops in range(args.extra_conv):
                    #print(f'B extra_conv_loops {extra_conv_loops} out_channels: {in_channels}')
                    # (T, J, D) => (T, J, D)
                    seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=in_channels,
                                            joint_num=self.edge_num[i], kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                                            stride=1,
                                            padding=padding, padding_mode=args.padding_mode, bias=bias))
                    seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
                # (T, J, D) => (T/2, J, 2D)
                #print(f'C out_channels: {out_channels}')
                seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=False,
                                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
                # self.convs.append(seq[-1])

                seq.append(pool)
                seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

        # in_features = self.channel_base[-1] * len(self.pooling_list[-1])
        # in_features *= int(args.temporal_scale / 2) 
        # self.reduce = nn.Linear(in_features, args.z_dim)
        # self.mu = nn.Linear(in_features, args.z_dim)
        # self.logvar = nn.Linear(in_features, args.z_dim)

    def forward(self, input):
        #bs, n, c = input.shape[0], input.shape[1], input.shape[2]
        output = input.permute(0, 2, 1)#input.reshape(bs, n, -1, 6)
        for layer in self.layers:
            output = layer(output)
        #output = output.view(output.shape[0], -1)
        output = output.permute(0, 2, 1)
        return output
class VQDecoderV3(nn.Module):
    def __init__(self, args):
        super(VQDecoderV3, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        args.vae_test_dim = args.joint_nums * args.eval_joint_dims
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            #print(f"{i}: ResBlock channels[0] {channels[0]}")
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        #print(f"VQDecoder inputs {inputs.shape}")
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
    
def getBoneId(bone_name):
    index = 0
    for key, value in data_tools.joints_list["spine_neck_141"].items():
        if bone_name == key:
            return index
        index += 1
    return -1

def readBoneStructure(path, joint_num):
    # 初始化一个空列表来存储父子关系对
    bone_pairs = []
    bone_pairs.append((0, joint_num))  # add an edge between the root joint and a virtual joint
    # 读取文件
    with open(path, 'r') as file:
        for line in file:
            # 去除行尾的换行符和空格，并分割成父子关节名
            parent, child = line.strip().split(',')
            # 使用getBoneId函数将关节名转换为ID
            parent_id = getBoneId(parent)
            child_id = getBoneId(child)
            # 如果转换成功（即ID不是-1），则添加到列表中
            if parent_id != -1 and child_id != -1:
                bone_pairs.append((parent_id, child_id))
            else:
                # 如果转换失败，可以选择打印错误消息或抛出异常
                print(f"Error: Unable to convert bone names to IDs for line: {line.strip()}")

    # 将列表转换为N*2的numpy数组
    bone_structure_array = np.array(bone_pairs)
    return bone_structure_array

#print(readBoneStructure(f"../spine_neck_141_boneid_structure"))