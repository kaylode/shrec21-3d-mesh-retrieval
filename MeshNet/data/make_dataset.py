import os
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root',
                    type=str,
                    help='path to MeshNet')
parser.add_argument('-t', '--task',
                    type=str,
                    help='task [Culture|Shape]')
parser.add_argument('-f', '--fold',
                    type=int,
                    help='fold index')
args = parser.parse_args()


if args.task == 'Shape':
    class_name = ['botella', 'cantaro', 'cuenco', 'figurina', 'lebrillo', 'olla', 'plato', 'vaso']
elif args.task == 'Culture':
    class_name = ['CHANCAY', 'LURIN', 'MARANGA', 'NAZCA', 'PANDO', 'SUPE']

def make_folder(df, type_='train'):
    class_indexes = df.class_id.unique()
    for class_id in class_indexes:
        class_path = os.path.join(ROOT_DIR, str(class_name[class_id]))
        if not os.path.exists(class_path):
            os.mkdir(class_path)
            os.mkdir(os.path.join(class_path,'train'))
            os.mkdir(os.path.join(class_path,'test'))

    class_count = [0 for i in range(len(class_indexes))]        
    for idx, row in tqdm(df.iterrows()):
        obj_id, class_id = row
        class_path = os.path.join(ROOT_DIR, str(class_name[class_id]), type_, f'{obj_id}.obj')
        copyfile(f"{args.root}/datasets/dataset{args.task}/objects/train/{obj_id}.obj", class_path)
        class_count[class_id] += 1
    print(class_count)




ROOT_DIR = f'{args.root}/datasets/dataset{args.task}/folds/fold_{args.fold}'
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
train_df = pd.read_csv(f'{args.root}/datasets/dataset{args.task}/annotations/{args.fold}_train.csv')
val_df = pd.read_csv(f'{args.root}/datasets/dataset{args.task}/annotations/{args.fold}_val.csv')


print('TRAINSET')
make_folder(train_df)

print('VALSET')
make_folder(val_df, type_ = 'test')