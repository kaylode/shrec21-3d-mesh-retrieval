import argparse

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def dotprod_dist(x, y):
    d = x.dot(y.T)
    d = (d - d.min(1)[:, np.newaxis]) / (d.max(1) - d.min(1))[:, np.newaxis]
    return 1 - d


def dist_func(mode):
    factory = {
        'euclidean': euclidean_distances,
        'cosine': cosine_distances,
        'dotprod': dotprod_dist,
    }
    if mode in factory:
        return factory[mode]
    else:
        raise Exception('Invalid mode.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query',
                        type=str,
                        help='path to the query embeddings')
    parser.add_argument('-g', '--gallery',
                        type=str,
                        help='path to the gallery embeddings')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='output file name and directory')
    parser.add_argument('-m', '--mode',
                        type=str,
                        help='distance [euclidean|cosine|dotprod]')
    parser.add_argument('-fmt', '--format',
                        type=str,
                        default='%10.6f',
                        help='printing format of each float')
    return parser.parse_args()


args = parse_args()

q = np.load(args.query)
g = np.load(args.gallery)
dist = dist_func(args.mode)

dist_mat = dist(q, g)
print(dist_mat.shape)

np.savetxt(args.output, dist_mat, fmt=args.format)
