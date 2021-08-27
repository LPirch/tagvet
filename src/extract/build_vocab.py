import re
import pickle
import argparse
from collections import Counter

import yaml
import joblib
from tqdm import tqdm

from features import gen_glogs, get_ref_hashes, get_tokens

RE_WORD = re.compile(r'\w+')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_zip', help='input zip file with glogs')
    parser.add_argument('label_file', help='label file with file hashes')
    parser.add_argument('vocab_file', help='pickle file for vocab')
    parser.add_argument('df_file', help='pickle file for document frequencies')
    parser.add_argument('--config_file', default='extraction_conf.yml', help='name of the config file')
    parser.add_argument('--jobs', type=int, default=1, help='number of parallel processes')
    return vars(parser.parse_args())


def load_yaml(in_file):
    return yaml.load(open(in_file), Loader=yaml.FullLoader)


def build_vocab(data_zip, label_file, jobs=1, **extraction_kwargs):
    """
    Build a vocabulary from glog reports and count the document frequency of each token.
    @param data_zip: input zip file with glogs
    @param label_file: label file with file hashes
    @param jobs: number of parallel processes to use
    @param extraction_kwargs: addition keyword/value arguments for the extraction
    """
    vocab = {'PAD': 0}
    df = Counter()
    ref_hashes = get_ref_hashes(label_file)
    print('[-] analyzing glogs')
    with joblib.parallel_backend('loky'):
        doc_frequencies = joblib.Parallel(n_jobs=jobs)(joblib.delayed(
            analyze_document)(data_zip, ref_h, extraction_kwargs) for ref_h in tqdm(ref_hashes))

    print('[-] merging document frequency counters and build vocab')
    df = Counter()
    for doc_freq in tqdm(doc_frequencies):
        df += doc_freq
    vocab = {}
    for key in df.keys():
        vocab[key] = len(vocab)
    return vocab, df


def analyze_document(data_zip, filehash, extraction_kwargs):
    """
    Count the token frequencies in a single document
    @param data_zip: input zip file with glogs
    @param filehash: hash of the malware, identifying the report
    @param extraction_kwargs: keyword/value arguments for extraction function
    """
    df = Counter()
    for glog in gen_glogs(data_zip, [filehash], get_tokens, verbose=False, **extraction_kwargs):
        glog_tokens = set()
        for tokens in glog:
            tokens = tokens.lower()
            glog_tokens |= set(RE_WORD.findall(tokens))
        for token in glog_tokens:
            if token not in df:
                df[token] = 1
    return df


def main(data_zip, label_file, vocab_file, df_file, config_file, jobs):
    """
    Build a vocabulary from glog reports, count the document frequency of each token and save both.
    @param data_zip: input zip file with glogs
    @param label_file: label file with file hashes
    @param vocab_file: output file for the vocabulary (pickle)
    @param df_file: output file for the document frequency (pickle)
    @param config_file: YAML file with extraction config
    @param jobs: number of parallel processes to use
    """
    cnf = load_yaml(config_file)
    vocab, df = build_vocab(data_zip, label_file, jobs, **cnf)
    with open(vocab_file, 'wb') as pkl:
        pickle.dump(vocab, pkl)
    with open(df_file, 'wb') as pkl:
        pickle.dump(df, pkl)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
