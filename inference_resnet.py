import os, sys
import numpy as np
from PIL import Image
import math
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet import *
from dataloader import *


def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k / sum_y for k in y]
    return z


def main():
    model_path = '/mnt/lustre/dingmingyu/kaggle/checkpoints/040_checkpoint.pth.tar'
    data_dir = '../data/datasets/'
    num_classes = 128
    new_height = 400
    new_width = 400
    batch_size = 512
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
    test_data = MyDataset(os.path.join(data_dir, 'test.txt'), data_dir, new_width, new_height, test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    params = torch.load(model_path)
    net = rgb_resnet152(pretrained=False, num_classes=num_classes)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(params['state_dict'])

    net.eval()
    print "Loading model ok..."

    pres = np.zeros((12800, 1))
    for i, (input, _, idx) in enumerate(test_loader):
        input = input.float().cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output = net(input_var)
        predict = output.data
        _, pred = predict.topk(1, 1, True, True)
        pred = pred.t()
        pred = pred.view(-1, 1)
        pred = pred.cpu().numpy()

        for j in range(pred.shape[0]):
            print idx[j]
            pres[int(idx[j]) - 1, :] = pred[j]

    csv_file = open("result_resnet152.csv", "w")
    writer = csv.writer(csv_file)
    file_header = ["id", "predicted"]
    writer.writerow(file_header)
    for i in range(12800):
        predict_label = pres[i] + 1
        row = [str(i + 1), str(int(predict_label))]
        writer.writerow(row)
    csv_file.close()

if __name__ == '__main__':
    main()