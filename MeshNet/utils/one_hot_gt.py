import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task',
                    type=str,
                    help='[Shape|Culture]')
parser.add_argument('-r', '--root',
                    type=str,
                    help='path to MeshNet')
parser.add_argument('-o', '--output_path',
                    type=str,
                    help='path to save gt .npy')
args = parser.parse_args()

CSV_FILE = f'{args.root}/datasets/dataset{args.task}/annotations/dataset.csv'

def one_hot_embedding(label):
    if args.task == 'Shape':
        num_classes = 8
    else:
        num_classes = 6
    return np.eye(num_classes)[label]

def main():
    embed_dict = {}
    embed_npy = []
    df = pd.read_csv(CSV_FILE)
    for idx, (obj_id, class_id) in df.iterrows():
        embed_dict[int(obj_id)] = one_hot_embedding(class_id)

    ids = sorted(list(embed_dict.keys()))
    for i in ids:
        embed_npy.append(embed_dict[i])

    print(f'Number of gallery embeddings: {len(ids)}')
    np.save(f'{args.output_path}',embed_npy)

if __name__ == '__main__':
    main()

