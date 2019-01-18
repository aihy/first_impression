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
parser.add_argument('--dir_path', default='/home/zihao_wang/demo/first-impression-v2/data/')
parser.add_argument('--arch', default='resnet50',
                    choices=['resnet18','resnet34','resnet50', 'resnet101', 'resnet152'])
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

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    train_data = MyDataset(os.path.join(args.dir_path, 'annotation_training.json'),
                           os.path.join(args.dir_path, 'train-jpglist.txt'),
                           os.path.join(args.dir_path, 'train-images'),
                           args.new_width,
                           args.new_length,
                           train_transform)
    val_data = MyDataset(os.path.join(args.dir_path, 'annotation_validation.json'),
                         os.path.join(args.dir_path, 'val-jpglist.txt'),
                         os.path.join(args.dir_path, 'val-images'),
                         args.new_width,
                         args.new_length,
                         val_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=True)

    if args.evaluate:
        validate(val_loader,model,criterion)
        return
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch: ' + str(epoch + 1))
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        #prec = validate(val_loader, model, criterion)

        # remember best prec and save checkpoint
        #is_best = prec > best_prec
        #best_prec = max(prec, best_prec)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            }, 0, checkpoint_name, args.resume)



def build_model():
    # model_name = "rgb_" + args.arch
    model_name = args.arch
    model = getattr(resnet, model_name)(pretrained=False, num_classes=1)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    return model


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda()
        input_var = torch.autograd.Variable(input)
        target = torch.cat(target,0)
        target = target.float().cuda()
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output, target_var, loss)
        losses.update(loss.data.item(), input.size(0))
        precs.update(prec)
    #    top1.update(prec[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {precs.val:.3f} ({precs.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, precs=precs))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    precs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.float()
        target = torch.cat(target,0)
        target = target.float().cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        loss.backward()
        # measure accuracy and record loss
        prec = accuracy(output, target_var, loss)
        losses.update(loss.data.item(), input.size(0))
        precs.update(prec)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {precs.val:.3f} ({precs.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                precs=precs))

    print(' * Prec {precs.avg:.3f} '
          .format(precs=precs))

    return precs.avg


def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)


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
    A=1-loss
    return A


if __name__ == '__main__':
    main()
