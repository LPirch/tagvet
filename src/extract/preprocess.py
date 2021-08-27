import re
import argparse
import pickle
import xml.etree.ElementTree as ET
from collections import Counter

from tqdm import tqdm
import joblib

from features import get_ref_hashes, _filter_xml_iter, get_tokens
from build_vocab import load_yaml


RE_DELIM = re.compile(r'(\W+)')
PAD_DIM = 1
PLACEHOLDER_DIM = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_zip', help='zip file with glog data')
    parser.add_argument('label_file', help='label file (csv)')
    parser.add_argument('vocab_file', help='token vocab (pickle)')
    parser.add_argument('feat_file', help='raw feature file (pickle)')
    parser.add_argument('vocab_out', help='output (word-level) vocab (pkl)')
    parser.add_argument('--config_file', default='extraction_conf.yml', help='name of the config file')
    parser.add_argument('--jobs', type=int, default=1, help='number of parallel processes')
    parser.add_argument('--min_df', type=int, default=1, help='minimum absolute document frequency')
    args = parser.parse_args()
    return vars(args)


def extract_features(zip_name, ref_hashes, vocab, feat_file, vocab_out, jobs, min_df, **extraction_kwargs):
    """
    Extract features from zipped glogs, construct vocabulary, embed features and save the feature matrix and vocabulary.
    NOTE: This uses a second vocabulary on the word level (combined tokens with delimiters like paths etc.) as opposed
          to the provided vocabulary on token level!
    @param zip_name: zip file with glog reports
    @param ref_hashes: reference list of hashes (for ordering)
    @param vocab: vocabulary dict
    @param feat_file: output file for feature matrix
    @param vocab_out: output file for new (filtered) vocab
    @param jobs: number of parallel processes
    @param min_df: minimum document frequency
    @param extraction_kwargs: additional extraction keyword/value arguments
    """
    print('[-] extract filtered raw data')
    with joblib.parallel_backend('loky'):
        X_raw = joblib.Parallel(n_jobs=jobs)(joblib.delayed(
            get_tokens)(zip_name, [ref_h], get_filtered, verbose=False, vocab=vocab, **extraction_kwargs)
            for ref_h in tqdm(ref_hashes))

    # analyze df
    print('[-] analyzing glogs')
    with joblib.parallel_backend('loky'):
        doc_freqs = joblib.Parallel(n_jobs=jobs)(joblib.delayed(
            count_df)(glog,) for glog in tqdm(X_raw))
    df = sum(doc_freqs, Counter())

    # build new vocab
    vocab = {'PAD': PAD_DIM, '*': PLACEHOLDER_DIM}
    max_row_len = 0
    print('[-] build new vocab')
    for glog in tqdm(X_raw):
        glog_tokens = set()
        for row in glog:
            glog_tokens |= set(row)
            max_row_len = max(len(row), max_row_len)
        for token in glog_tokens:
            if token not in vocab and df[token] >= min_df:
                vocab[token] = len(vocab) + 1
    print("[-] finished. vocab size: ", len(vocab))

    # embed
    print('[-] embedding dataset')
    with joblib.parallel_backend('loky'):
        X_emb = joblib.Parallel(n_jobs=jobs)(joblib.delayed(
            embed_single)(glog, vocab, max_row_len) for glog in tqdm(X_raw))

    # save
    with open(feat_file, 'wb') as pkl:
        pickle.dump(X_emb, pkl)
    with open(vocab_out, 'wb') as pkl:
        pickle.dump(vocab, pkl)


def get_filtered(glog, vocab=None, ignored_nodes=None, ignored_attribs=None):
    """ Retrieve filtered tokens from glog contents (as string) """
    xml = ET.fromstring(glog)
    for node in _filter_xml_iter(xml, ignored_nodes, ignored_attribs):
        yield _serialize_vocab(node, vocab)


def _serialize_vocab(node, vocab):
    """
    Replace rare substrings by '*' and merge them again using the original delimiter.
    @param node: xml node (glog function call)
    @param vocab: dictionary mapping tokens to dimensions
    """
    tokens = []
    tag = node.tag
    if node.tag not in vocab:
        tag = '*'
    tokens.append(tag)

    for k, v in node.attrib.items():
        v = v.lower()
        if k not in vocab:
            k = '*'
        tokens.append(k)
        split_v = RE_DELIM.split(v)
        delims = []
        vals = []
        for sp_val in split_v:
            if RE_DELIM.match(sp_val):
                delims.append(sp_val)
            else:
                vals.append(sp_val)
        # add one delim at the end for equally-sized lists
        delims.append('')
        vals = [v if v in vocab else '*' for v in vals]
        # merge with delimiters
        merged_vals = []
        for val, delim in zip(vals, delims):
            merged_vals.append(val)
            merged_vals.append(delim)
        tokens.append(''.join(merged_vals))
    return tokens


def count_df(glog):
    """
    Count document frequencies.
    @param glog: xml root of glog.xml
    """
    df = Counter()
    glog_tokens = set()
    for tokens in glog:
        glog_tokens |= set(tokens)
    for token in glog_tokens:
        if token not in df:
            df[token] = 1
    return df


def embed_single(glog, vocab, max_row_len):
    """
    Embed glog tokens according to a vocabulary, clipping at max_row_len.
    @param glog: list of extracted tokens (after _serialize_vocab)
    @param vocab: dictionary mapping each token to a dimension
    @param max_row_len: maximum number of function call arguments (determined by analyzing the dataset
                        and used to pad all rows to the same length)
    """
    glog_rows = []
    for row in glog:
        emb_row = [vocab.get(token, PLACEHOLDER_DIM) for token in row]
        emb_row += [PAD_DIM]*(max_row_len - len(emb_row))
        glog_rows.append(emb_row)
    return glog_rows


def main(data_zip, label_file, vocab_file, feat_file, vocab_out, jobs, config_file, min_df):
    """
    Embed features according to a given vocabulary and save them in a pickle file.
    @param data_zip: zip file with glog data
    @param label_file: label file (csv)
    @param vocab_file: token vocab (pickle)
    @param feat_file: raw feature file (pickle)
    @param vocab_out: output (word-level) vocab (pkl)
    @param config_file: name of the config file
    @param jobs: number of parallel processes
    @param min_df: minimum absolute document frequency
    """
    cnf = load_yaml(config_file)
    ref_hashes = get_ref_hashes(label_file)
    vocab = pickle.load(open(vocab_file, 'rb'))
    extract_features(data_zip, ref_hashes, vocab, feat_file, vocab_out, jobs, min_df, **cnf)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
