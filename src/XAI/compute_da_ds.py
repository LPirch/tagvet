import os
import argparse
import pickle as pkl

import numpy as np
from tqdm import tqdm

from DNN_training import get_damd_cnn


def compute_da(args):
    step_size, max_dims_to_delete = 50, 2000
    replacement_token_idx = 1  # PAD
    data = pkl.load(open(args.data_path, 'rb'))
    model_w_softmax = get_damd_cnn(args.no_tokens, args.no_classes, final_nonlinearity=args.nonlinearity)
    model_w_softmax.load_weights(args.model_path)
    # compute initial classes of samples and compute initial scores
    print('Predicting samples ...')
    dims_to_explain, softmax_scores = [], []
    x_axis, y_axis = [], []
    for sample in tqdm(data):
        s_arr = np.array(sample).reshape(1, -1)
        prediction = model_w_softmax.predict(s_arr)[0]
        dims_to_explain.append(np.argmax(prediction))
        softmax_scores.append(np.max(prediction))
    x_axis.append(0)
    y_axis.append(np.mean(softmax_scores))
    print('Average value in beginning:{}'.format(y_axis[0]))
    filenames_pkl = os.listdir(args.data_folder)
    filename_ordered = [name + '.pkl' for name in open(args.path_filenames).read().splitlines()]
    # save order of dimensions to delete
    if args.random:
        relevances_argsort = [np.random.choice(list(range(len(xx))), max_dims_to_delete, replace=False) for xx in data]
    else:
        print('Calculating order of deletion')
        relevances_argsort = []
        for fn, max_idx, data_sample in tqdm(zip(filename_ordered, dims_to_explain, data), total=len(filename_ordered)):
            if fn in filenames_pkl:
                fp = os.path.join(args.data_folder, fn)
                rel_vectors = pkl.load(open(fp, 'rb'))
                rel_vector = rel_vectors[max_idx]
                if type(rel_vector) is not np.ndarray:
                    print('wrong index for relevances of fn {}'.format(fn))
                    relevances_argsort.append([])
                else:
                    assert len(rel_vector) == len(data_sample)
                    rel_vector_idx_sorted = np.argsort(rel_vector)[::-1]  # biggest to smallest
                    relevances_argsort.append(rel_vector_idx_sorted[:max_dims_to_delete])
            else:
                print('Did not find file for {}'.format(fn))
                return -1
    # predict samples again without the most relevant tokens
    for k in range(step_size, max_dims_to_delete, step_size):
        softmax_scores = []
        print('Deleting {} features'.format(k))
        for sample, rel_argsorted, orig_class in tqdm(zip(data, relevances_argsort, dims_to_explain), total=len(data)):
            if len(rel_argsorted) == 0:
                continue
            s_arr = np.array(sample).reshape(1, -1)
            indices_to_remove = rel_argsorted[:k]
            s_arr[0, indices_to_remove] = replacement_token_idx
            prediction = model_w_softmax.predict(s_arr)[0]
            softmax_scores.append(prediction[orig_class])
        print('Average value after deletion of {} features: {}'.format(k, np.mean(softmax_scores)))
        x_axis.append(k)
        y_axis.append(np.mean(softmax_scores))
    save_path = os.path.join(args.save_path, 'results.da')
    with open(save_path, 'a') as f:
        print('model: {}'.format(args.model_path), file=f)
        print('no_deleted_values: {}'.format(','.join([str(x) for x in x_axis])), file=f)
        print('average_softmax_score: {}'.format(','.join([str(y) for y in y_axis])), file=f)


def compute_ds(args):
    data = pkl.load(open(args.data_path, 'rb'))
    model_w_softmax = get_damd_cnn(args.no_tokens, args.no_classes, final_nonlinearity=args.nonlinearity)
    model_w_softmax.load_weights(args.model_path)
    filenames_pkl = os.listdir(args.data_folder)
    filename_ordered = [name + '.pkl' for name in open(args.path_filenames).read().splitlines()]
    # compute initial classes of samples and compute initial scores
    print('Predicting samples ...')
    dims_w_explanation = []
    for sample in tqdm(data):
        s_arr = np.array(sample).reshape(1, -1)
        prediction = model_w_softmax.predict(s_arr)[0]
        dims_w_explanation.append(np.argmax(prediction))
    # collect relevance values
    all_rels = []
    for fn, max_idx, data_sample in tqdm(zip(filename_ordered, dims_w_explanation, data), total=len(filename_ordered)):
        if fn in filenames_pkl:
            fp = os.path.join(args.data_folder, fn)
            rel_vectors = pkl.load(open(fp, 'rb'))
            rel_vector = rel_vectors[max_idx]
            if type(rel_vector) is not np.ndarray:
                print('wrong index for relevances of fn {}'.format(fn))
            else:
                assert len(rel_vector) == len(data_sample)
                abs_max = np.max(np.abs(rel_vector))
                rel_vector *= 1./abs_max
                all_rels += [r for r in rel_vector if r != 0]
        else:
            print('Did not find file for {}'.format(fn))
            return -1
    values, bins = np.histogram(all_rels, bins=100)
    # compute ds
    ds = [0]
    # list of uneven length has unique middle point
    if len(values) % 2 == 1:
        no_steps = len(values) - 1
        mid = int(no_steps / 2)
        x_axis = [0] + [bins[mid+i] for i in range(mid)]
        for i in range(mid):
            ds.append(ds[i] + values[mid - i] + values[mid + i])
    # else, use two middle points as starting point
    else:
        no_steps = len(values) / 2
        mid_high = int(len(values) / 2)
        mid_low = mid_high - 1
        x_axis = [bins[mid_high+i] for i in range(int(no_steps)+1)]
        for i in range(int(no_steps)):
            ds.append(ds[i] + values[mid_low - i] + values[mid_high + i])
    final_ds_curve = np.array(ds) * 1. / np.sum(values)
    print(final_ds_curve)
    print(len(final_ds_curve))
    print(x_axis)
    print(len(x_axis))
    save_path = os.path.join(args.save_path, 'results.ds')
    with open(save_path, 'a') as f:
        print('model: {}'.format(args.model_path), file=f)
        print('x_axis: {}'.format(','.join([str(x) for x in x_axis])), file=f)
        print('ds_score: {}'.format(','.join([str(y) for y in final_ds_curve])), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    # compute da
    da_parser = subparsers.add_parser('compute_da')
    da_parser.add_argument('model_path', type=str, help='Path to model as hdf5 file')
    da_parser.add_argument('data_path', type=str, help='Path to list representation of glogs')
    da_parser.add_argument('data_folder', type=str, help='Path to folder containing pkls with relevances')
    da_parser.add_argument('path_filenames', type=str, help='Path to file containing ordered filenames')
    da_parser.add_argument('no_classes', type=int, help='Number of classes that can be predicted')
    da_parser.add_argument('no_tokens', type=int, help='Total number of tokens that appear in data')
    da_parser.add_argument('save_path', type=str, help='Save folder for results')
    da_parser.add_argument('--nonlinearity', type=str, help='Final nonlinearity in network', default='softmax')
    # compute ds
    ds_parser = subparsers.add_parser('compute_ds')
    ds_parser.add_argument('model_path', type=str, help='Path to model as hdf5 file')
    ds_parser.add_argument('data_path', type=str, help='Path to list representation of glogs')
    ds_parser.add_argument('data_folder', type=str, help='Path to folder containing pkls with relevances')
    ds_parser.add_argument('path_filenames', type=str, help='Path to file containing ordered filenames')
    ds_parser.add_argument('no_classes', type=int, help='Number of classes that can be predicted')
    ds_parser.add_argument('no_tokens', type=int, help='Total number of tokens that appear in data')
    ds_parser.add_argument('save_path', type=str, help='Save folder for results')
    ds_parser.add_argument('--nonlinearity', type=str, help='Final nonlinearity in network', default='softmax')
    ds_parser.add_argument('--random', help='Choose relevances randomly (baseline approach)', action='store_true')

    args = parser.parse_args()
    if args.command == 'compute_da':
        compute_da(args)
    elif args.command == 'compute_ds':
        compute_ds(args)
