import copy
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import SHREC21Dataset
from models import MeshNet
from tqdm import tqdm
from losses import FocalLoss
from metrics import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--data_root',
                    type=str,
                    help='path to MeshNet')
parser.add_argument('-t', '--task',
                    type=str,
                    help='[Culture|Shape]')
parser.add_argument('-s', '--saved_path',
                    type=str,
                    help='save checkpoint to')
parser.add_argument('--num_faces',
                    type=int,
                    help='number of faces')
args = parser.parse_args()

cfg = get_train_config()
cfg['dataset']['data_root'] = args.data_root
cfg['dataset']['max_faces'] = args.num_faces
cfg['saved_path'] = args.saved_path

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = {
    x: SHREC21Dataset(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=8, shuffle=True, pin_memory=True, collate_fn=data_set[x].collate_fn)
    for x in ['train', 'test']
}


def train_model(model, metrics, criterion, optimizer, scheduler, cfg):
    best_values = {
        "acc":0.0,
        "bl_acc": 0.0,
        "f1-score": 0.0,
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    try:
        for epoch in range(1, cfg['max_epoch']):

            print('-' * 60)
            print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
            print('-' * 60)

            for phrase in ['train', 'test']:

                if phrase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for i, (centers, corners, normals, neighbor_index, targets) in enumerate(tqdm(data_loader[phrase])):

                    optimizer.zero_grad()

                    centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
                    corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
                    normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
                    neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
                    targets = Variable(torch.cuda.LongTensor(targets.cuda()))

                    with torch.set_grad_enabled(phrase == 'train'):
                        outputs, _ = model(centers, corners, normals, neighbor_index)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)

                        if phrase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:
                            for metric in metrics:
                                metric.update(outputs, targets)

                        running_loss += loss.item() * centers.size(0)
                        running_corrects += torch.sum(preds == targets.data)

                epoch_loss = running_loss / len(data_set[phrase])
                epoch_acc = running_corrects.double() / len(data_set[phrase])

                if phrase == 'train':
                    scheduler.step()
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))

                if phrase == 'test':
                    metric_dict = {}
                    for metric in metrics:
                        metric_dict.update(metric.value())
                        metric.reset()
                    
                    for key, value in metric_dict.items():
                        print(key, ': ', value)

                    for key, value in metric_dict.items():
                        if key not in ["each_acc"]:
                            if metric_dict[key] > best_values[key]:
                                best_values[key] = metric_dict[key]
                                value = np.round(float(best_values[key]), 4)
                                best_model_wts = copy.deepcopy(model.state_dict())
                                torch.save(best_model_wts, os.path.join(cfg['saved_path'], f'MeshNet_best_{key}.pkl'))
                        
                    print('{} Loss: {:.4f}'.format(phrase, epoch_loss))

    except KeyboardInterrupt:
        return best_values

    return best_values

if __name__ == '__main__':

    if args.task == 'Shape':
        num_classes = 8
    else:
        num_classes = 6

    model = MeshNet(cfg=cfg['MeshNet'], num_classes=num_classes, require_fea=True)
    model = nn.DataParallel(model)

    if 'pretrained' in cfg.keys():
        model.load_state_dict(torch.load(cfg['pretrained']))

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    metrics = [
        AccuracyMetric(), 
        BalancedAccuracyMetric(num_classes=num_classes),
        F1ScoreMetric(n_classes=num_classes), ]

    if not os.path.exists(cfg['saved_path']):
        os.mkdir(cfg['saved_path'])

    best_values = train_model(model, metrics, criterion, optimizer, scheduler, cfg)
    
    for k,v in best_values.items():
        print(f"Best {k}: {v}")
    
        
