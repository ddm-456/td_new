import os
import yaml
import sys
import torch
import torch.utils.data as data
import cv2
import os.path as osp
import time
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchutil import AverageMeter, create_logger
import torch.optim as optim
import random
import h5py
import re
import water
from test import test


from math import exp
from data_loader_augmentation import ICDAR2015, Synth80k, ICDAR2013

###import file#######
from mseloss import Maploss



from collections import OrderedDict
from eval.script import getresult



from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool
import os
#3.2768e-5
random.seed(42)

def mkdirs(dir):
    if not osp.exists(dir):
        os.mkdir(dir)

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")



# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT reimplementation')


    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--batch_size', default=128, type = int,
                        help='batch size of training')
    #parser.add_argument('--cdua', default=True, type=str2bool,
                        #help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=32, type=int,
                        help='Number of workers used in dataloading')

    parser.add_argument('--config', type=str, default='cfgs/synth_exp001.yaml')
    parser.add_argument('--trained_model', default='./craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')


    args = parser.parse_args()


    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/" + args.exp_name))
    mkdirs(osp.join("checkpoint", args.exp_name))
    mkdirs(osp.join("checkpoint", args.exp_name, "result"))

    logger = create_logger('global_logger', "logs/" + args.exp_name + '/log.txt')
    logger.info('{}'.format(args))

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))



    # gaussian = gaussion_transform()
    # box = scio.loadmat('/data/CRAFT-pytorch/syntext/SynthText/gt.mat')
    # bbox = box['wordBB'][0][0][0]
    # charbox = box['charBB'][0]
    # imgname = box['imnames'][0]
    # imgtxt = box['txt'][0]

    #dataloader = syndata(imgname, charbox, imgtxt)
    dataloader = Synth80k('./data/SynthText', target_size = args.target_size)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    #batch_syn = iter(train_loader)
    # prefetcher = data_prefetcher(dataloader)
    # input, target1, target2 = prefetcher.next()
    #print(input.size())
    net = CRAFT()
    #net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/CRAFT_net_050000.pth')))
    #net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/1-7.pth')))
    #net.load_state_dict(copyStateDict(torch.load('/data/CRAFT-pytorch/craft_mlt_25k.pth')))
    #net.load_state_dict(copyStateDict(torch.load('vgg16_bn-6c64b313.pth')))
    #realdata = realdata(net)
    # realdata = ICDAR2015(net, '/data/CRAFT-pytorch/icdar2015', target_size = 768)
    # real_data_loader = torch.utils.data.DataLoader(
    #     realdata,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)
    net = net.cuda()
    #net = CRAFT_net

    # if args.cdua:
    net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True
    # realdata = ICDAR2015(net, '/data/CRAFT-pytorch/icdar2015', target_size=768)
    # real_data_loader = torch.utils.data.DataLoader(
    #     realdata,
    #     batch_size=10,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)


    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    net.train()


    step_index = 0


    loss_time = 0
    loss_value = 0
    compare_loss = 1

    batch_time = AverageMeter(100)
    iter_time = AverageMeter(100)

    loss_value = AverageMeter(10)
    args.max_iters = args.num_epoch * len(train_loader)

    for epoch in range(args.num_epoch):
        # if epoch % 50 == 0 and epoch != 0:
        #     step_index += 1
        #     adjust_learning_rate(optimizer, args.gamma, step_index)

        for index, (images, gh_label, gah_label, mask, _) in enumerate(train_loader):

            st = time.time()
            index = epoch*len(train_loader) + index
            if index % 10000 == 0 and index != 0:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            #real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
            idx = index + epoch * int(len(train_loader) / args.batch_size)

            # syn_images, syn_gh_label, syn_gah_label, syn_mask = next(batch_syn)
            # images = torch.cat((syn_images,real_images), 0)
            # gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
            # gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
            # mask = torch.cat((syn_mask, real_mask), 0)

            #affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)


            images = Variable(images.type(torch.FloatTensor)).cuda()
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            gah_label = Variable(gah_label).cuda()
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()
            batch_time.update(time.time() - st)
            # affinity_mask = affinity_mask.type(torch.FloatTensor)
            # affinity_mask = Variable(affinity_mask).cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            loss_value.update(loss.item())
            iter_time.update(time.time() - st)


            remain_iter = args.max_iters - (idx + epoch * int(len(train_loader)/args.batch_size))
            remain_time = remain_iter * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)

            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if index % args.print_freq == 0:
                logger.info('Iter = [{0}/{1}]\t'
                                'data time = {batch_time.avg:.3f}\t'
                                'iter time = {iter_time.avg:.3f}\t'
                                'loss = {loss.avg:.4f}\t'.format(
                                    idx, args.max_iters, batch_time=batch_time,
                                    iter_time=iter_time,
                                    loss=loss_value))

                logger.info("remain_time: {}".format(remain_time))


            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth'

            if index % args.eval_iter== 0 and index != 0:
                print('Saving state, index:', index)
                torch.save(net.module.state_dict(),
                           './checkpoint/{}/synweights_'.format(args.exp_name) + repr(index) + '.pth')
                test('./checkpoint/{}/synweights_'.format(args.exp_name) + repr(index) + '.pth', args=args,
                     result_folder='./checkpoint/{}/result/'.format(args.exp_name))
                #test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
                res_dict = getresult('./checkpoint/{}/result/'.format(args.exp_name))
                logger.info(res_dict['method'])








