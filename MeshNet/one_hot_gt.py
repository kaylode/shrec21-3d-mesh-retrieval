import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd



CSV_FILE = '/home/nhtlong/pmkhoi/shrec21/retrieval/MeshNet/datasets/datasetShape/annotations/dataset.csv'

def one_hot_embedding(label):
    return np.eye(8)[label]

def main():
    embed_dict = {}
    embed_npy = []
    df = pd.read_csv(CSV_FILE)
    for idx, (obj_id, class_id) in df.iterrows():
        embed_dict[int(obj_id)] = one_hot_embedding(class_id)

    ids = sorted(list(embed_dict.keys()))
    for i in ids:
        embed_npy.append(embed_dict[i])

    print(f'Number of embeddings: {len(ids)}')
    np.save('./results/gt.npy',embed_npy)

if __name__ == '__main__':
    main()

