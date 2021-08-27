import os
import json
import argparse

import numpy as np
import pickle as pkl
from tqdm import tqdm
import innvestigate

from DNN_training import get_damd_cnn


def explain_behavior(analyzer, data, labels, dims_to_explain, filenames, save_path):
    """
    Creates explanation for each data instance
    @param analyzer: Object of innvestigate analyzer class
    @param data: List of lists representing sequences of glog calls as indices
    @param labels: Labels corresponding to data
    @param dims_to_explain: List of classes for which explanation shall be generated
    @param filenames: Name of each sample that is used as identifier when saving results
    @param save_path: Location where to store results
    """
    no_behaviors = labels.shape[1]
    if not os.path.isdir(os.path.join(save_path, 'relevances_raw')):
        os.makedirs(os.path.join(save_path, 'relevances_raw'))
    for d, dimensions, fname in tqdm(zip(data, dims_to_explain, filenames), total=len(data)):
        d = np.array(d).reshape(1, -1)
        no_glog_calls = d.shape[1]
        rel = []
        for i in range(no_behaviors):
            if i in dimensions:
                r = list(analyzer.analyze(d, neuron_selection=i).reshape((no_glog_calls,)))
                rel.append(r)
            else:
                rel.append(np.nan)
        # these can become really large in sum, therefore we save each separatly
        filename = os.path.join(save_path, 'relevances_raw', fname) + '.pkl'
        pkl.dump(rel, open(filename, 'wb'))


def get_explanations_for_behavior(filenames, data, idx2call, idx2behave, save_path, index_range=10, top_n=10):
    """
    Summarizes raw explanations by using the 'top_n' tokens with highest relevance in each sample and saves the
    surrounding of the tokens where surrounding is given by 'index_range' functions calls before and after the token
    @param filenames: Name of each sample that is used as identifier when saving results
    @param data: List of lists representing sequences of glog calls as indices
    @param idx2call: Dictionary that maps an index in data to the corresponding function call
    @param idx2behave: Dictionary that maps an index in labels to the corresponding name
    @param save_path: Path where raw relevances as produced by 'explain_behavior' lie
    @param index_range: How many indices before and after the most relevant tokens shall be considered
    @param top_n: How many of the most relevant tokens shall be analyzed
    """
    if not os.path.isdir(os.path.join(save_path, 'relevances_text')):
        os.makedirs(os.path.join(save_path, 'relevances_text'))
    for filename, d in tqdm(zip(filenames, data), total=len(data)):
        json_dict = {}
        rel_vector = pkl.load(open(os.path.join(save_path, 'relevances_raw', filename+'.pkl'), 'rb'))
        for i in range(rel_vector.shape[0]):
            if not(np.isnan(rel_vector[i, :]).all()):
                max_relevance = np.max(np.abs(rel_vector[i]))
                behavior_name = idx2behave[i]
                json_dict[behavior_name] = {}
                top_n = np.argsort(-rel_vector[i, :])[:top_n]
                json_dict[behavior_name]['top-10-tokens'] = {}
                json_dict[behavior_name]['surroundings'] = {}
                for idx_no, idx in enumerate(top_n):
                    token_name = idx2call[d[idx]]
                    token_precessor = idx2call[d[idx-1]] if idx > 0 else ''
                    token_decessor = idx2call[d[idx + 1]] if idx != len(d)-1 else ''
                    running_idx = idx
                    fcall_name = 'not-found'
                    # search for preceeding function call (they start with "gfn")
                    while fcall_name == 'not-found' and running_idx > 0:
                        preceeding_token_name = idx2call[d[running_idx]]
                        if preceeding_token_name[:3] == 'gfn':
                            fcall_name = preceeding_token_name
                        running_idx -= 1
                    token_name = f'[{fcall_name}][{token_precessor}]{token_name}[{token_decessor}]'
                    rel_value = rel_vector[i, idx]
                    rel_value_normed = 1./max_relevance * rel_value if max_relevance != 0 else 0
                    json_dict[behavior_name]['top-10-tokens'][token_name] = rel_value_normed
                    json_dict[behavior_name]['surroundings'][token_name] = []
                    range_lower = idx - index_range if idx >= index_range else 0
                    range_upper = idx + index_range if idx <= len(d) - index_range else len(d)
                    for glog_idx in range(range_lower, range_upper):
                        token_name_surr = idx2call[d[glog_idx]]
                        rel_surr = rel_vector[i, glog_idx]
                        rel_surr_normed = 1./max_relevance*rel_vector[i, glog_idx] if rel_surr != 0 else 0
                        json_dict[behavior_name]['surroundings'][token_name].append([token_name_surr, rel_surr_normed])
            with open(os.path.join(save_path, 'relevances_text', filename+'.json'), 'w') as f:
                json.dump(json_dict, f, indent=4)


def gen_explanations_args(args):
    data = pkl.load(open(args.data_path, 'rb'))
    labels = np.load(args.label_path)
    filenames = open(args.filename_path).read().splitlines()
    calls = open(args.glog_call_path).read().splitlines()
    no_labels = labels.shape[1]
    no_tokens = len(calls)
    print('no tokens', no_tokens)
    model_w_softmax = get_damd_cnn(no_tokens, no_labels, final_nonlinearity=args.nonlinearity)
    model_wo_softmax = get_damd_cnn(no_tokens, no_labels, final_nonlinearity=None)
    model_w_softmax.load_weights(args.model_path)
    model_wo_softmax.load_weights(args.model_path)
    if args.calculate_raw:
        print('Predicting samples ...')
        dims_to_explain = []
        for sample in tqdm(data):
            s_arr = np.array(sample).reshape(1, -1)
            prediction = model_w_softmax.predict(s_arr)
            if args.nonlinearity == 'softmax':
                dims_to_explain.append([np.argmax(prediction[0])])
            else:
                dims_to_explain.append(np.where(prediction > 0.5)[1])
        analyzer = innvestigate.create_analyzer('lrp.epsilon', model_wo_softmax,
                                                neuron_selection_mode='index', epsilon=1e-2)
    tag_names = open(args.tag_names).read().splitlines() if args.tag_names is not None else None
    idx_to_call = dict(zip(range(1, len(calls) + 1), calls))
    idx_2_tag = dict(zip(range(len(tag_names)), tag_names)) if tag_names is not None \
        else dict(zip(range(no_labels), [str(x) for x in range(no_labels)]))
    if args.calculate_raw:
        explain_behavior(analyzer, data, labels, dims_to_explain, filenames, args.save_path)
    get_explanations_for_behavior(filenames, data, idx_to_call, idx_2_tag, args.save_path)


def average_explanations(args):
    save_folder = args.save_folder
    filenames = [f for f in os.listdir(args.data_dir) if f.endswith('json')]
    filepaths = [os.path.join(args.data_dir, f) for f in filenames]
    behavior_dict = {}
    print('Averaging {} reports'.format(len(filepaths)))
    for fp in tqdm(filepaths):
        sample_dict = json.load(open(fp, 'r'))
        for behavior in sample_dict:
            if behavior not in behavior_dict:
                behavior_dict[behavior] = {}
                behavior_dict[behavior]['occurences'] = 1
            else:
                behavior_dict[behavior]['occurences'] += 1
            for feature in sample_dict[behavior]['top-10-tokens']:
                feature_relevance = sample_dict[behavior]['top-10-tokens'][feature]
                # order of entries : occurence, percentage_in_top_10, relevances, mean_relevance
                if feature not in behavior_dict[behavior]:
                    behavior_dict[behavior][feature] = {}
                    behavior_dict[behavior][feature]['no_occurences'] = 1
                    behavior_dict[behavior][feature]['avg_relevance'] = feature_relevance
                    behavior_dict[behavior][feature]['relevances'] = [feature_relevance]
                else:
                    behavior_dict[behavior][feature]['no_occurences'] += 1
                    behavior_dict[behavior][feature]['relevances'].append(feature_relevance)
                    behavior_dict[behavior][feature]['avg_relevance'] = np.mean(
                        behavior_dict[behavior][feature]['relevances'])
    # update relative frequencies
    for behavior in behavior_dict:
        for feature in behavior_dict[behavior]:
            if feature == 'occurences':
                continue
            behavior_occ = behavior_dict[behavior]['occurences']
            feature_occ = behavior_dict[behavior][feature]['no_occurences']
            behavior_dict[behavior][feature]['percentage_in_top_10'] = feature_occ / behavior_occ
    # for each behavior save json
    for behavior in behavior_dict:
        behavior_dict_ = behavior_dict[behavior]
        for feature in behavior_dict_:
            if feature == 'occurences':
                continue
            del behavior_dict_[feature]['relevances']
        with open(os.path.join(save_folder, behavior+'.json'), 'w') as f:
            json.dump(behavior_dict_, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates explanations for Trained Model.')
    subparsers = parser.add_subparsers(dest="command")
    # generate explanations
    expl_parser = subparsers.add_parser("generate_explanations",
                                        help="Creates explanations and saves them in json file")
    expl_parser.add_argument("model_path", type=str, help="Path to tensorflow model to explain (as .hdf5 file)")
    expl_parser.add_argument("data_path", type=str, help="Path to data to explain (as .pkl file)")
    expl_parser.add_argument("label_path", type=str, help="Path to labels belonging to data (as .npy file)")
    expl_parser.add_argument("glog_call_path", type=str, help="Path to file containing all glog function calls")
    expl_parser.add_argument("save_path", type=str, help="Where to save explanations and results")
    expl_parser.add_argument("filename_path", type=str, help="File containing filenames for each data sample")
    expl_parser.add_argument("--tag_names", type=str, help="Path to file containing names for tags")
    expl_parser.add_argument("--nonlinearity", type=str, help="Final nonlinearity used for model", default='softmax')
    expl_parser.add_argument("--calculate_raw", help="Whether to calculate raw explanations aswell. The raw"
                                                     "explanations can use a lot of disk space and are normally"
                                                     "not necessary", action='store_true')

    # average_explanations
    train_parser = subparsers.add_parser("average_explanations", help="Averages explanations in directory.")
    train_parser.add_argument("data_dir", type=str, help="Path to folder containing jsons as generated by"
                                                         "'generate_explanations' call")
    train_parser.add_argument("save_folder", type=str, help="Path to save destination of averaged analyses")
    args = parser.parse_args()
    if args.command == 'generate_explanations':
        gen_explanations_args(args)
    elif args.command == 'average_explanations':
        average_explanations(args)
