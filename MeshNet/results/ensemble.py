import os
import numpy as np
import glob as glob
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--npy_folder',
                    type=str,
                    help='path to folder contains test embedding .npy files on all folds')

args = parser.parse_args()


def ensemble(args):
    files = glob.glob(os.path.join(args.npy_folder, '*.npy'))
    prob=0
    for file in tqdm(files):
        prob_ = np.load(file, allow_pickle=True)
        prob += prob_
    prob/=len(files)

    np.save(args.npy_folder+'/ensemble.npy', prob, allow_pickle=True)

if __name__ =='__main__':
    ensemble(args)
