import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from models import MeshNet
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight',
                    type=str,
                    help='path to the trained weight')
parser.add_argument('-t', '--task',
                    type=str,
                    help='[Culture|Shape]')
args = parser.parse_args()


class Testset(data.Dataset):

    def __init__(self, cfg):
        self.root = cfg['test_root']
        self.augment_data = cfg['augment_data']
        self.max_faces = cfg['max_faces']

        self.data = []
        for filename in os.listdir(self.root):
            filepath = os.path.join(self.root, filename)
            if filename.endswith('.npz'):
                self.data.append(filepath)

    def __getitem__(self, i):
        path = self.data[i]
        data = np.load(path)
        face = data['faces']
        neighbor_index = data['neighbors']

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        
        filename = os.path.basename(path)
        return centers, corners, normals, neighbor_index, filename 

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        centers = torch.stack([i[0] if i[0].shape[-1] == self.max_faces   else F.pad(i[0], pad=(0, 1), mode='constant', value=0) for i in batch])
        corners = torch.stack([i[1] if i[1].shape[-1] == self.max_faces else F.pad(i[1], pad=(0, 1), mode='constant', value=0) for i in batch])
        normals = torch.stack([i[2] if i[2].shape[-1] == self.max_faces else F.pad(i[2], pad=(0, 1), mode='constant', value=0) for i in batch])
        neighbor_index = torch.stack([i[3] if i[3].shape[0] == self.max_faces else torch.cat([i[3], torch.zeros(1,3)]) for i in batch]).type(torch.LongTensor)
        filename = [i[5] for i in batch]
        return centers, corners, normals, neighbor_index, filename

cfg = get_test_config()
cfg['dataset']['test_root'] = args.test_root
cfg['dataset']['max_faces'] = args.num_faces

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']

data_set = Testset(cfg=cfg['dataset'])
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=4, shuffle=True, pin_memory=True)


def inference(model):
    embed_dict = {}
    embed_npy = []
    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, filename) in enumerate(tqdm(data_loader)):
            file_id = int(filename[0][:-4])
            centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
            corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
            normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
            neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

            _, feas = model(centers, corners, normals, neighbor_index)
            ft_all = feas.cpu().squeeze(0).numpy()
            embed_dict[file_id] = ft_all

    ids = sorted(list(embed_dict.keys()))
    for i in ids:
        embed_npy.append(embed_dict[i])

    print(f'Number of embeddings in testset: {len(ids)}')
    np.save(f'./results/test/embed.npy',embed_npy)

if __name__ == '__main__':
    if args.task == 'Shape':
        num_classes = 8
    else:
        num_classes = 6

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.cuda()
    model = nn.DataParallel(model)
    model.module.classifier[-1] = nn.Linear(in_features=256, out_features=num_classes).cuda()
    model.load_state_dict(torch.load(args.weight))

    if not os.path.exists(f'./results/test'):
        os.makedirs(f'./results/test')
    model.eval()

    inference(model)
