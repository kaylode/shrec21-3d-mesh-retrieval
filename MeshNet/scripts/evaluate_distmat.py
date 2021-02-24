import argparse

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query',
                        type=str,
                        help='path to the query csv')
    parser.add_argument('-g', '--gallery',
                        type=str,
                        help='path to the gallery csv')
    parser.add_argument('-d', '--distmat',
                        type=str,
                        help='path to distance matrix txt')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='output file name and directory')
    return parser.parse_args()


def get_retrieved_labels(g_mapping, dist_vec):
    order_list = np.argsort(dist_vec)
    return np.array([
        g_mapping[x]
        for x in order_list
        if x in g_mapping
    ])


def nearest_neighbor(q_label, retrieved_labels):
    return int(q_label == retrieved_labels[0])


def first_tier(q_label, retrieved_labels):
    n_relevant_objs = (retrieved_labels == q_label).sum()
    retrieved_1st_tier = retrieved_labels[:n_relevant_objs]
    return (retrieved_1st_tier == q_label).mean()


def second_tier(q_label, retrieved_labels):
    n_relevant_objs = (retrieved_labels == q_label).sum()
    retrieved_2nd_tier = retrieved_labels[:2*n_relevant_objs]
    return (retrieved_2nd_tier == q_label).mean() * 2


def mean_average_precision(q_label, retrieved_labels):
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(retrieved_labels):
        if p == q_label and p not in retrieved_labels[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score


# Parse arguments
args = parse_args()

# Generate mapping of gallery ids to their corresponding label
g_df = pd.read_csv(args.gallery)
g_mapping = {
    id: label
    for id, label in zip(g_df['obj_id'].values, g_df['class_id'].values)
}

# Load dataset and distance matrix
q_df = pd.read_csv(args.query)

# Load predicted distance matrix
dist_mtx = np.loadtxt(args.distmat)

# Define metrics
METRICS = ['MAP', 'NN', 'FT', 'ST']

metric_compute = {
    'NN': nearest_neighbor,
    'FT': first_tier,
    'ST': second_tier,
    'MAP': mean_average_precision,
}

# Compute for each query
score = {
    metric_id: [
        metric_compute[metric_id](
            q_label,
            get_retrieved_labels(g_mapping, dist_mtx[qid])
        )
        for qid, q_label in zip(q_df['obj_id'].values, q_df['class_id'].values)
    ]
    for metric_id in METRICS
}

# Output to dataframe
q_df = pd.concat([q_df, pd.DataFrame(score)], axis=1)

# Save to file
q_df.to_csv(args.output, index=False)

# Print summary
print(q_df[METRICS].mean())
print(q_df.groupby('class_id')[METRICS].mean())
