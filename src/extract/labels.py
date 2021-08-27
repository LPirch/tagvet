import os
import argparse
from zipfile import ZipFile

from features import sort_files, get_ref_hashes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('zip_name', help='input zip')
    parser.add_argument('hash_file', help='file with hash list')
    parser.add_argument('out_file', help='output file (csv)')
    args = parser.parse_args()
    return vars(args)


def get_labels(zip_name, hash_file):
    """
    Read hashes and malware families from the file names in a zip file.
    Expected naming: [.*]/[family]/[hash]/glog.xml
    @param zip_name: zip file containing a set of glog.xml files
    @hash_file: file with a reference list of hashes (for ordering)
    """
    with ZipFile(zip_name, 'r') as z:
        filenames = z.namelist()
    hashes = get_ref_hashes(hash_file)
    filenames = sort_files(filenames, hashes)
    families = [f.split(os.sep)[-3] for f in filenames]
    return hashes, families


def main(zip_name, hash_file, out_file):
    """
    Read hashes and labels from a zip file and store them in a csv file.
    @param zip_name: input zip
    @param hash_file: file with hash list
    @param out_file: output file (csv)
    """
    hashes, labels = get_labels(zip_name, hash_file)
    with open(out_file, 'w') as f:
        for h, label in zip(hashes, labels):
            f.write('{},{}\n'.format(h, label))


if __name__ == '__main__':
    args = parse_args()
    main(**args)
