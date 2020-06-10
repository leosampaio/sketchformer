"""
create_token_dict.py
Created on Oct 03 2019 11:39
Build a dictionary of sketch tokens
@author: Tu Bui tb0035@surrey.ac.uk
"""
import os
import sys
import numpy as np
import argparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import inspect
cdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, '/'.join(cdir.split('/')[:-2]))  # up 2 levels
import utils
from utils.helpers import Timer, save_pickle, load_pickle


def load_data(files):
    """
    load all stroke3 data into 2 giant dxdy arrays (each has size Nx2)
    the first array contains points with penstate=0
    -   2nd              -               penstate=1
    :param files: list of files with stroke3 data (.npy)
    :return: two arrays, each with data for one pen state
    """
    out1, out2 = [], []
    for i, path in enumerate(files):
        print("Loading {} ({}/{})".format(path, i, len(files)))
        data = np.load(path, encoding='latin1', allow_pickle=True)
        samples = data['train']
        for sketch in samples:
            sketch = normalize_sketch(sketch)
            pen_lift_ids = np.where(sketch[:, 2] == 1)[0] + 1  # offset to get next touch point
            pen_lift_ids = pen_lift_ids[:-1]  # exclude end of sketch
            pen_hold_ids = set(range(len(sketch))) - set(pen_lift_ids)

            out1.append(sketch[list(pen_hold_ids), :2])
            out2.append(sketch[list(pen_lift_ids), :2])
    return np.concatenate(out1), np.concatenate(out2)


def normalize_sketch(sketch):
    # removes large gaps from the data
    sketch = np.minimum(sketch, 1000)
    sketch = np.maximum(sketch, -1000)

    # get bounds of sketch and use them to normalise
    min_x, max_x, min_y, max_y = utils.sketch.get_bounds(sketch)
    max_dim = max([max_x - min_x, max_y - min_y, 1])
    sketch = sketch.astype(np.float32)
    sketch[:, :2] /= max_dim
    return sketch


if __name__ == '__main__':

    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir')
    parser.add_argument('-s', '--vocab-size', default=1000, type=int)
    parser.add_argument('--n-samples', default=5000000, type=int)
    parser.add_argument('-m', '--method', default='k-means')
    parser.add_argument('-r', '--p1-ratio', default=0.2, type=float,
                        help="Ratio of points with penstate=1 (minority) vs penstate=0 (majority) in SAMPLES; set to 0 if disable")
    parser.add_argument('--class-list', type=str, default='prep_data/quickdraw/list_quickdraw.txt')
    parser.add_argument('--target-file', type=str, default='prep_data/sketch_token/token_dict.pkl')

    args = parser.parse_args()

    class_names = []
    with open(args.class_list) as clf:
        class_names = clf.read().splitlines()

    class_files = []
    for class_name in class_names:
        file = "{}/{}.npz".format(args.dataset_dir, class_name)
        class_files.append(file)

    timer = Timer()
    args = parser.parse_args()
    print("Loading data ...")
    data_p0, data_p1 = load_data(class_files)
    N_P0, N_P1 = data_p0.shape[0], data_p1.shape[0]
    print("p1/p0 natural ratio: %f" % (N_P1 / N_P0))
    if args.p1_ratio > 0 and args.method == 'k-means':
        n_p1 = int(args.p1_ratio * args.num_samples)
        n_p0 = args.num_samples - n_p1
        if N_P0 > n_p0:
            print("Sample %d out of %d points with penstate 0" % (n_p0, N_P0))
            ids_p0 = np.random.choice(N_P0, n_p0, replace=False)
            data_p0 = data_p0[ids_p0]
        if N_P1 > n_p1:
            print("Sample %d out of %d points with penstate 1" % (n_p1, N_P1))
            ids_p1 = np.random.choice(N_P1, n_p1, replace=False)
            data_p1 = data_p1[ids_p1]
    data = np.r_[data_p0, data_p1]

    N = data.shape[0]
    print("Loading data done, took {}".format(timer.time(True)))
    print("Building dictionary ...")
    if args.method == 'mini-batch-k-means':
        cluster = MiniBatchKMeans(n_clusters=args.vocab_size, max_iter=200, compute_labels=False,
                                  batch_size=2**16, verbose=0, n_init=5, init_size=2**18).fit(data)
    elif args.method == 'k-means':
        cluster = KMeans(n_clusters=args.vocab_size, n_init=10, max_iter=500, tol=1e-6, n_jobs=10,
                         verbose=0).fit(data)
    else:
        print('Unsupported clustering method: %s' % args.method)
        sys.exit(1)
    print("Dictionary built: {}".format(timer.time(True)))
    save_pickle(args.target_file, cluster)
    print("Total time: {}".format(timer.time(False)))
