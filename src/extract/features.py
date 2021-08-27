import argparse
import os
import io
import pickle
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_zip', help='zip file with glog data')
    parser.add_argument('hash_file', help='file with reference hashes (in order)')
    parser.add_argument('ngram_len', type=int, help='length of ngrams')
    parser.add_argument('min_df', type=int, help='minimum absolute doc frequency')
    parser.add_argument('max_df', type=float, help='maximum relative doc frequency')
    parser.add_argument('--binary', action='store_true', help='whether to extract binary vectors')
    parser.add_argument('out_file', help='output file (pickle)')
    args = parser.parse_args()
    return vars(args)


def get_ref_hashes(in_file):
    """
    Read list of reference hashes from a file. Used to ensure the sames processing order across experiments.
    @param in_file: csv or simple text file containing a hash in each line (at first position for csv)
    """
    if in_file.endswith('.csv'):
        with open(in_file, 'r') as f:
            return [line.strip().split(',')[0] for line in f.read().splitlines()]
    else:
        with open(in_file, 'r') as f:
            return [line.strip() for line in f.read().splitlines()]


def sort_files(filenames, ref_hashes):
    """
    Sort files according to a list of reference hashes.
    @param filenames: list of file names to be sorted
    @param ref_hashes: list of reference hashes (of the malware)
    """
    filenames = [f for f in filenames if 'glog.xml' in f]
    hashes = [f.split(os.sep)[-2] for f in filenames]
    idx = [hashes.index(h) for i, h in enumerate(ref_hashes)]
    filenames = [filenames[i] for i in idx]
    return filenames


def gen_glogs(zipname, ref_hashes, extraction, verbose=True, **extraction_kwargs):
    """
    Yield extracted features from a set of glog reports stored in a zip file.
    @param zipname: name of the zip file containing all glog reports
    @param ref_hashes: list of malware hashes to determine the order of processed samples
    @param extraction: function extracting features from a single glog
    @param verbose: enable/disable tqdm progress bar
    @param extraction_kwargs: additional keyword/value arguments passed to extraction function
    """
    with ZipFile(zipname, 'r') as z:
        # sort files according to reference hash list
        filenames = sort_files(z.namelist(), ref_hashes)
        if verbose:
            filenames = tqdm(filenames)

        # load and return glog
        for filename in filenames:
            with io.BytesIO(z.read(filename)) as buffer:
                glog = buffer.getvalue()
                yield extraction(glog, **extraction_kwargs)


def get_fncalls(glog, ignored_nodes=None, ignored_attribs=None):
    """
    Get space-separated list of function calls from glog content (as string).
    @param glog: glog.xml content as string
    @param ignored_nodes: Iterable containing node xml tags to ignore
    @param ignored_attribs: Iterable containning node xml attribute keys to ignore
    """

    xml = ET.fromstring(glog)
    fncalls = [node.tag for node in _filter_xml_iter(xml, ignored_nodes, ignored_attribs)]
    return ' '.join(fncalls)


def get_tokens(glog, ignored_nodes=None, ignored_attribs=None):
    """
    Yield serialized function call nodes from a glog xml report, optionally ignoring certain nodes/attributes.
    @param glog: glog.xml content as string
    @param ignored_nodes: Iterable containing node xml tags to ignore
    @param ignored_attribs: Iterable containning node xml attribute keys to ignore
    """
    xml = ET.fromstring(glog)
    for node in _filter_xml_iter(xml, ignored_nodes, ignored_attribs):
        yield _serialize(node)


def _filter_xml_iter(xml, ignored_nodes, ignored_attribs):
    """
    Yield gfn nodes from glog xml report, optionally filtering out nodes and attribs.
    @param xml: xml root node (ElementTree object)
    @param ignored_nodes: Iterable containing node xml tags to ignore
    @param ignored_attribs: Iterable containning node xml attribute keys to ignore
    """
    for node in xml.iter():
        if node.tag.startswith('gfn_') and node.tag not in ignored_nodes:
            attribs = _pull_child_attribs(node)
            for attr, val in list(attribs.items()):
                if attr in ignored_attribs:
                    del attribs[attr]
                elif val.startswith('e_id_') or attr.endswith('_obj'):
                    del attribs[attr]
                elif 'address' in attr:
                    del attribs[attr]
                elif attr == 'handle' and val.startswith('0x'):
                    del attribs[attr]
            yield node


def _pull_child_attribs(node):
    """
    Pull xml attributes of child nodes and merge them into the attrib dict of the current node.
    The glog xml report function call nodes may contain children (the objects on which they operate)
    but their attributes a) dont collide with parent attribs and b) often contain relevant information.
    We therefore inline this information in the parent xml node.
    @param node: xml node to return attributes from
    """
    attr = node.attrib
    for child in node:
        attr.update(child.attrib)
    return attr


def _serialize(node):
    """
    Serialize a xml node tag and its attribute dict in a reproducible way.
    @param node: xml node to serialize
    """
    tokens = [node.tag]
    for k, v in node.attrib.items():
        tokens.extend([k, v])
    return ' '.join(tokens)


def main(data_zip, hash_file, ngram_len, min_df, max_df, binary, out_file):
    """
    Extract tf-idf-weighted n-gram features from a set of glog files in a zip file.
    @param data_zip: zip file with glog data
    @param hash_file: file with reference hashes (in order)
    @param ngram_len: length of ngrams
    @param min_df: minimum absolute doc frequency
    @param max_df: maximum relative doc frequency
    @param binary: whether to extract binary vectors
    @param out_file: output file (pickle)
    """
    vec = TfidfVectorizer(ngram_range=(ngram_len, ngram_len), binary=binary, min_df=min_df, max_df=max_df)
    X = vec.fit_transform(gen_glogs(data_zip, get_ref_hashes(hash_file)), get_fncalls)
    with open(out_file, 'wb') as pkl:
        pickle.dump((X, vec.vocabulary_), pkl)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
