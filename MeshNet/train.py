import copy
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map
from tqdm import tqdm
from losses import FocalLoss
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
    x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=8, shuffle=True, pin_memory=True, collate_fn=data_set[x].collate_fn)
    for x in ['train', 'test']
}


def train_model(model, criterion, optimizer, scheduler, cfg):

    best_acc = 0.0
    best_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    try:
        for epoch in range(1, cfg['max_epoch']):

            print('-' * 60)
            print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
            print('-' * 60)

            for phrase in ['train', 'test']:

                if phrase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                ft_all, lbl_all = None, None

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


                        running_loss += loss.item() * centers.size(0)
                        running_corrects += torch.sum(preds == targets.data)

                epoch_loss = running_loss / len(data_set[phrase])
                epoch_acc = running_corrects.double() / len(data_set[phrase])

                if phrase == 'train':
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))

                if phrase == 'test':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if epoch % 10 == 0:
                        torch.save(copy.deepcopy(model.state_dict()), os.path.join(cfg['saved_path'],f'{epoch}.pkl'))

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))
    except KeyboardInterrupt:
        return best_model_wts, best_acc

    return best_model_wts, best_acc


if __name__ == '__main__':

    if args.task == 'Shape':
        num_classes = 8
    else:
        num_classes = 6

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model = nn.DataParallel(model)

    if 'pretrained' in cfg.keys():
        model.load_state_dict(torch.load(cfg['pretrained']))

    model.module.classifier[-1] = nn.Linear(in_features=256, out_features=num_classes)
    model.cuda()

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])

    
    best_model_wts, best_acc = train_model(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts, os.path.join(cfg['saved_path'], f'MeshNet_best_{best_acc}.pkl'))
    print(f'Best model saved! Best Acc: {best_acc}')
    
        
