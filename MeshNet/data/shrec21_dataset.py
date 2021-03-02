import numpy as np
import os
import torch
import torch.utils.data as data
import torch.nn.functional as F

# type_to_index_map = {
#     'botella':0, 
#     'cantaro':1,
#     'cuenco':2,
#     'figurina':3,
#     'lebrillo':4,
#     'olla':5,
#     'plato':6,
#     'vaso':7,
# }

type_to_index_map = {
    'CHANCAY':0, 
    'LURIN':1,
    'MARANGA':2,
    'NAZCA':3,
    'PANDO':4,
    'SUPE':5
}

class SHREC21Dataset(data.Dataset):

    def __init__(self, cfg, part='train', return_index=False):
        self.root = cfg['data_root']
        self.augment_data = cfg['augment_data']
        self.max_faces = cfg['max_faces']
        self.part = part
        self.return_index = return_index

        self.data = []
        for type in os.listdir(self.root):
            type_index = type_to_index_map[type]
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), type_index))

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['faces']
        neighbor_index = data['neighbors']

        # data augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

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
        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        if self.return_index:
            filename = os.path.basename(path)
            return centers, corners, normals, neighbor_index, target, filename 
        else:
            return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)

    

    def collate_fn(self, batch):
        centers = torch.stack([i[0] if i[0].shape[-1] == self.max_faces   else F.pad(i[0], pad=(0, 1), mode='constant', value=0) for i in batch])
        corners = torch.stack([i[1] if i[1].shape[-1] == self.max_faces else F.pad(i[1], pad=(0, 1), mode='constant', value=0) for i in batch])
        normals = torch.stack([i[2] if i[2].shape[-1] == self.max_faces else F.pad(i[2], pad=(0, 1), mode='constant', value=0) for i in batch])
        neighbor_index = torch.stack([i[3] if i[3].shape[0] == self.max_faces else torch.cat([i[3], torch.zeros(1,3)]) for i in batch]).type(torch.LongTensor)
        target = torch.stack([i[4] for i in batch])

        if self.return_index:
            filename = [i[5] for i in batch]
            return centers, corners, normals, neighbor_index, target, filename
        else:
            return centers, corners, normals, neighbor_index, target

