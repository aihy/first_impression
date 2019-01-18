import argparse
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms

import resnet
from dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir_path', default='/home/zihao_wang/demo/first-impression-v2/data/')
parser.add_argument('--arch', default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=200, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--iter-size', default=4, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 8)')
parser.add_argument('--new_length', default=400, type=int)
parser.add_argument('--new_width', default=400, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 200)')
parser.add_argument('--resume', default='../checkpoints', type=str, metavar='PATH',
                    help='path to checkpoints (default: ../checkpoints)')
parser.add_argument('--resumepath', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0


def main():
    global args, best_prec
    args = parser.parse_args()
    print("Build model ...")
    model = build_model()
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    if args.resumepath:
        if os.path.isfile(args.resumepath):
            print("=> loading checkpoint '{}'".format(args.resumepath))
            checkpoint = torch.load(args.resumepath)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(args.evaluate, checkpoint['epoch'], best_prec))
        else:
            print("=> no checkpoint found at '{}'".format(args.resumepath))
            return

    cudnn.benchmark = True

    # data transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    test_data = MyDataset(os.path.join(args.dir_path, 'xixi'),
                          os.path.join(args.dir_path, 'test-jpglist.txt'),
                          os.path.join(args.dir_path, 'test-images'),
                          args.new_width,
                          args.new_length,
                          val_transform)

    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    test(test_loader, model, criterion)
    return


def personality_test(input):
    print("Build model ...")
    model = build_model()
    checkpoint = torch.load('personality_checkpoint.pth.tar')
    # args.start_epoch = checkpoint['epoch']
    best_prec = checkpoint['best_prec']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, best_prec {})"
          .format(checkpoint['epoch'], best_prec))
    
    cudnn.benchmark = True

    # data transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    test_data = MyDataset(os.path.join('data', 'xixi'),
                          os.path.join('data', 'test-jpglist.txt'),
                          os.path.join('data', 'images'),
                          args.new_width,
                          args.new_length,
                          val_transform)

    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    test(test_loader, model, criterion)
    return


def build_model():
    model = getattr(resnet, resnet18)(pretrained=False, num_classes=1)
    model = torch.nn.DataParallel(model).cuda()
    return model


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    precs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    print("extraversion\tneuroticism\tagreeableness\tconscientiousness\topenness")
    print("外向性\t神经质\t亲和性\t尽责性\t经验开放性")
    for i, (input, target) in enumerate(test_loader):
        input = input.float()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)

        # compute output
        output = model(input_var)
        print(output)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(
                      i, len(test_loader), batch_time=batch_time,))

    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['lr'] = param_group['lr']/2


def accuracy(output, target, loss):
    A = 1-loss
    return A


if __name__ == '__main__':
    main()
