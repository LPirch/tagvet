# TagVet - Vetting Malware Tags using Explainable Machine Learning

The is the repo for [our paper](https://intellisec.de/pubs/2021-eurosec.pdf) "TagVet - Vetting Malware Tags using Explainable Machine Learning" (Proceedings of the 14th European Workshop on Systems Security. 2021).

## About The Project

TagVet uses XAI methods to explain black-box labeling processes on malware traces.
It is able to trace back characteristic features for unsupervised clusterings, supervised predictions, external taggings and any other potential label source.
In summary, we use the following procedure:
1. preprocessing: see below
2. feature extraction: sequences of tf-idf weighted tokens (fixed vocabulary embedding)
3. gather a) cluster IDs (unsupervised), b) classification predictions (supervised), c) external references (tags)
4. train surrogate model (CNN) to predict the labels from a), b) and c)
5. apply XAI methods on each surrogate model to explain the respective black-box labeling

### Dataset

We used a custom dataset with about 5000 malware reports from the [VMRay Analyzer Sandbox](https://www.vmray.com/products/analyzer-malware-sandbox/)
with behavioral tags inferred by the sandbox as well as malware family labels obtained from [VirusTotal](https://www.virustotal.com/gui/home/upload).
Unfortunately, the dataset and the precise format cannot be shared but we provide a general overview below, in [our paper](https://intellisec.de/pubs/2021-eurosec.pdf) and our [presentation](https://www.youtube.com/watch?v=uryY68EF3wg).

Each report comprises a sequence of function calls and their arguments.
The analyzed malware targets the Windows OS and therefore the function calls correspond to Windows API calls in our case.
The basic structure of a function call in XML-like syntax is
```XML
<function_name some_id="XYZ" some_timestamp="1234" arg0="val0" argN="valN"/>
```

### Preprocessing

Preprocessing is done to reduce the noise in malware-controlled fields of the function call traces (e.g. path names) and to retain valuable information.
We hence split each value at common delimiters, count the occurrences of single tokens and replace rare ones with a placeholder token.
For example, the paths
```
C:\Windows\System32\A\common.dll
C:\Windows\System32\B\common.dll
C:\Windows\System32\C\xyz.dll
```
will become
```
C:\Windows\System32\*\common.dll
C:\Windows\System32\*\common.dll
C:\Windows\System32\*\*.dll
```
during preprocessing.
This greatly reduces the number of distinct strings and allows us to embed them using a fixed vocabulary.

## Installation

To get started, clone the repo and install the python dependencies. We suggest using a virtual environment for that:
```sh
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

After the initial setup, the general workflow looks as follows (in pseudo-bash):

```sh
GLOG_ZIP=/path/to/glog.zip      # zip file with glog reports (/[family]/[hash]/glog.xml)
LABEL_FILE=/path/to/labels.txt  # list of malware labels
HASH_FILE=/path/to/hashes.txt   # list of malware hashes to process the data in a well-defined ordering
CACHE_DIR=/path/to/local/cache  # used for storing intermediate information
MODEL_DIR=/path/to/models
RESULT_DIR=/path/to/results

# some general configs
MIN_DF=5
N_CORES=1
source venv/bin/activate

# extract features
pushd src/extract
python build_vocab.py ${GLOG_ZIP} ${LABEL_FILE} ${CACHE_DIR}/token_vocab.pkl ${CACHE_DIR}/token_df.pkl
python filter_vocab.py ${CACHE_DIR}/token_vocab.pkl ${CACHE_DIR}/token_df.pkl ${LABEL_FILE} $MIN_DF ${CACHE_DIR}/token_vocab_filtered.pkl
python preprocess.py ${GLOG_ZIP} ${LABEL_FILE} ${CACHE_DIR}/token_vocab_filtered.pkl ${CACHE_DIR}/emb_features.pkl ${CACHE_DIR}/word_vocab.pkl
popd

# perform clustering evaluation
pushd src/cluster
python cluster.py ${CACHE_DIR}/emb_features.pkl ${CACHE_DIR}/features.pkl ${LABEL_FILE} ${RESULT_DIR}/clustering_results.json ${RESULT_DIR}/clustering_prediction.json $N_CORES
popd

# train surrogate model + XAI evaluation
pushd src/XAI
FCALL_FILE=/path/to/fcalls.txt                  # (see docs in code)
LABELS_TO_NAMES=/path/to/labels_to_names.txt    # (see docs in code)
python DNN_training.py ${CACHE_DIR}/emb_features.pkl ${LABEL_FILE} /path/to/fcall.txt --save_path ${MODEL_DIR} --labels_to_names ${LABELS_TO_NAMES}
python explanation_generation.py ${MODEL_DIR}/some_model.hdf5 ${CACHE_DIR}/emb_features.pkl ${LABEL_FILE} [...]
python compute_da_ds.py [compute_da|compute_ds] ${MODEL_DIR}/some_model.hdf5 ${CACHE_DIR}/emb_features.pkl
popd
```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Lukas Pirch - [lukpirch at tu-bs.de](mailto:lukpirch@tu-bs.de).