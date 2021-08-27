import pickle
import argparse

from features import get_ref_hashes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_file', help='pickle file for vocab')
    parser.add_argument('df_file', help='pickle file for document frequencies')
    parser.add_argument('label_file', help='csv file with labels (used to infer n_samples)')
    parser.add_argument('min_df', type=float, help='minimum document frequency')
    parser.add_argument('out_file', help='output pickle file (new vocab)')
    return vars(parser.parse_args())


def filter_vocab(vocab, df, min_df):
    """
    Filter out rare tokens and construct a new vocabulary.
    @param vocab: input vocabulary
    @param df: document frequency dictionary (term: frequency)
    @param min_df: minimum document frequency
    """
    new_vocab = {'PAD': 0}
    for term in vocab:
        if term != 'PAD' and df[term] >= min_df:
            new_vocab[term] = len(new_vocab)
    return new_vocab


def main(vocab_file, df_file, label_file, min_df, out_file):
    """
    Remove rare tokens from existing vocabulary. This script is useful to try out different vocabulary sizes without
    having to re-analyze the whole corpus.
    @param vocab_file: pickle file for vocab
    @param df_file: pickle file for document frequencies
    @param label_file: csv file with labels (used to infer n_samples)
    @param min_df: minimum document frequency
    @param out_file: output pickle file (new vocab)
    """
    vocab = pickle.load(open(vocab_file, 'rb'))
    df = pickle.load(open(df_file, 'rb'))
    hashes = get_ref_hashes(label_file)
    min_df *= len(hashes)  # min_df is relative to number of samples

    print("old vocabulary size: ", len(vocab))
    vocab = filter_vocab(vocab, df, min_df)
    print("new vocabulary size: ", len(vocab))
    with open(out_file, 'wb') as pkl:
        pickle.dump(vocab, pkl)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
