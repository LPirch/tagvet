import os
import argparse
import pickle as pkl

import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D, Dropout
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.utils import shuffle
import tensorflow as tf


def get_damd_cnn(no_tokens, no_labels, final_nonlinearity='softmax', kernel_size_1=21, kernel_size_2=5):
    """
    Get CNN model as used in the experiments.
    @param no_tokens: Number of tokens appearing in data
    @param no_labels: Number of classes that can be predicted
    @param final_nonlinearity: Final nonlinearity in the network
    @param kernel_size_1: Kernel size of first convolutional layer (default=21)
    @param kernel_size_2: Kernel size of second convolutional layer (default=5)
    """
    embedding_dimensions = 128
    number_of_dense_units = 64
    model = Sequential()
    model.add(Embedding(no_tokens+1, output_dim=embedding_dimensions))
    model.add(Conv1D(filters=64, kernel_size=kernel_size_1, strides=kernel_size_1, padding='valid', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=kernel_size_2, strides=2, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(number_of_dense_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_labels, activation=final_nonlinearity))
    # print(model.summary())
    return model


def train_network_batchwise(data, labels, network, no_epochs, batch_size, testset_size, save_path='models',
                            random_state=42, labels_to_names=None, loss='categorical_crossentropy'):
    """
    Trains the CNN model in batchwise manner.
    @param data: List of lists where each lists is sequence of functions calls represented by indices
    @param labels: Numpy array of same len as data indicating classes of corresponding data sample
    @param network: Tensorflow cnn model as produced by 'get_damd_cnn'
    @param no_epochs: Number of epochs to train the model
    @param batch_size: Batch size used in training
    @param testset_size: Size of the testset (will be generated out of 'data')
    @param save_path: Path to location where the final model will be saved
    @param random_state: Random state that can be set for reproducible results
    @param labels_to_names: Optional dict that maps class indices to names
    @param loss: String for loss function that will be optimized during training
        """
    if loss == 'binary_crossentropy':
        L = binary_crossentropy
    else:
        L = categorical_crossentropy
    log_fp = os.path.join(save_path, 'Train_log')
    network.compile(optimizer='adam', loss=L, metrics=['accuracy'])
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=testset_size,
                                                                        random_state=random_state)
    data_valid, data_test, labels_valid, labels_test = train_test_split(data_test, labels_test, test_size=0.5,
                                                                        random_state=random_state)
    weights = []
    best_model_idx, best_aucpr = 0, 0
    print('Training for {} epochs'.format(no_epochs))
    for j in tqdm(range(no_epochs)):
        for i in range(0, len(data_train), batch_size):
            data_arr = data_train[i:i+batch_size]
            label_batch = labels_train[i:i+batch_size]
            grads = []
            for sample, label in zip(data_arr, label_batch):
                sample_arr = np.array(sample).reshape(1, -1)
                with tf.GradientTape() as tape:
                    result = network(sample_arr)
                    loss = network.loss(label.reshape(1, -1), result)
                    grad = tape.gradient(loss, network.trainable_weights)
                for le in range(len(grad)):
                    if type(grad[le]) == tf.IndexedSlices:
                        grad[le] = tf.convert_to_tensor(grad[le])
                if len(grads) == 0:
                    grads = [g/batch_size for g in grad]
                else:
                    for le in range(len(grad)):
                        grads[le] += grad[le]/batch_size
            network.optimizer.apply_gradients(zip(grads, network.trainable_weights))
        weights.append(network.get_weights())
        with open(log_fp, 'a') as f:
            print('\n Validation performance epoch {}:\n'.format(j), file=f)
        tpr, fpr, acc, auc, f1, aucpr = eval_model(
            network, data_valid, labels_valid, save_path, label_file_path=labels_to_names)
        if aucpr > best_aucpr:
            print('New best aucpr: {}'.format(aucpr))
            best_aucpr = aucpr
            best_model_idx = j
        network.save_weights(os.path.join(save_path, 'damd_model_{}.hdf5'.format(j)))
        data_train, labels_train = shuffle(data_train, labels_train)
    # finally use best model on test data
    print('Evaluating on test data:')
    network.set_weights(weights[best_model_idx])
    with open(log_fp, 'a') as f:
        print('\n Test performance:\n', file=f)
    eval_model(network, data_test, labels_test, save_path, label_file_path=labels_to_names)


def eval_model(model, test_data, test_labels, save_path, label_file_path=None):
    """
    Evaluate model on given data
    @param model: Tensorflow model for evaluation
    @param test_data: Test data - list of lists with instructions as indices
    @param test_labels: Labels corresponding to 'test data'
    @param save_path: Path to location to store results
    @param label_file_path: Path to file that contains string names of each class, i.e. line n is name for class n
    @return:
    """
    preds, test_losses = [], []
    n_test_samples = test_labels.shape[0]
    n_occurences = np.sum(test_labels, axis=0)
    class_weights = 1./n_test_samples * n_occurences
    for i in range(0, len(test_data)):
        data_batch = np.array(test_data[i]).reshape(1, -1)
        label_batch = np.array(test_labels[i]).reshape(1, -1)
        preds.append(model.predict_on_batch(data_batch))
        test_losses.append(model.evaluate(data_batch, label_batch, verbose=0))
    print('Test loss: {}'.format(np.mean(test_losses)))
    preds = np.vstack(preds)
    tprs, fprs, accs, rocs, F1s, aucprs = [], [], [], [], [], []
    not_enough = 0
    for i in range(preds.shape[1]):
        preds_i = preds[:, i]
        preds_i_rounded = np.zeros(preds_i.shape)
        preds_i_rounded[preds_i > 0.5] = 1
        labels_i = test_labels[:, i]
        cm = confusion_matrix(labels_i, preds_i_rounded)
        if len(np.unique(labels_i)) > 1:  # and cm.shape[0] > 1 and cm.shape[1] > 1:
            TN, FN, TP, FP = cm[0, 0], cm[1, 0], cm[1, 1], cm[0, 1]
            FPR = FP / (FP + TN)
            TPR = TP / (TP + FN)
            ACC = (TP+TN)/(TP+TN+FP+FN)
            F1 = f1_score(labels_i, preds_i_rounded)
            roc_auc = roc_auc_score(labels_i, preds_i)
            precision, recall, _ = precision_recall_curve(labels_i, preds_i)
            prauc = auc(recall, precision)
            tprs.append(TPR)
            fprs.append(FPR)
            accs.append(ACC)
            rocs.append(roc_auc)
            F1s.append(F1)
            aucprs.append(prauc)
        else:
            print('Not enough data for feature {}'.format(i))
            not_enough += 1
            tprs.append(np.NaN)
            fprs.append(np.NaN)
            accs.append(np.NaN)
            rocs.append(np.NaN)
            F1s.append(np.NaN)
            aucprs.append(np.NaN)
    if label_file_path is not None:
        behavior = open(label_file_path).read().splitlines()
    else:
        behavior = ['tag_id{}'.format(i) for i in range(preds.shape[1])]
    log_fp = os.path.join(save_path, 'Train_log')
    with open(log_fp, 'a') as f:
        for i, b in enumerate(behavior):
            print('{}: ACC: {} TPR: {} FPR: {} AUC:{} F1: {} AUCPR: {}'.format(
                  b, accs[i], tprs[i], fprs[i], rocs[i], F1s[i], aucprs[i]), file=f)
        print('\n'+'='*90+'\n', file=f)
        weighted_tpr = np.where(np.isnan(tprs), 0, tprs).dot(class_weights)
        weighted_fpr = np.where(np.isnan(fprs), 0, fprs).dot(class_weights)
        weighted_acc = np.where(np.isnan(accs), 0, accs).dot(class_weights)
        weighted_roc_auc = np.where(np.isnan(rocs), 0, rocs).dot(class_weights)
        weighted_f1 = np.where(np.isnan(F1s), 0, F1s).dot(class_weights)
        weighted_aucpr = np.where(np.isnan(aucprs), 0, aucprs).dot(class_weights)
        print('Mean TPR: {} Median TPR: {}'.format(weighted_tpr, np.nanmedian(tprs)), file=f)
        print('Mean FPR: {} Median FPR: {}:'.format(weighted_fpr, np.nanmedian(fprs)), file=f)
        print('Mean ACC: {} Median ACC: {}:'.format(weighted_acc, np.nanmedian(accs)), file=f)
        print('Mean AUC: {} Median AUC: {}:'.format(weighted_roc_auc, np.nanmedian(roc_auc)), file=f)
        print('Mean F1: {} Median F1: {}:'.format(weighted_f1, np.nanmedian(F1s)), file=f)
        print('Mean AUCPR: {} Median AUCPR: {}:'.format(weighted_aucpr, np.nanmedian(aucprs)), file=f)
    return weighted_tpr, weighted_fpr, weighted_acc, weighted_roc_auc, weighted_f1, weighted_aucpr


def train_model_args(args):
    data = pkl.load(open(args.data_path, 'rb'))
    labels = np.load(args.label_path)
    all_tokens = open(args.fcall_path).readlines()
    no_tokens = len(all_tokens)
    no_labels = labels.shape[1]
    print('{} data samples'.format(len(data)))
    print('{} label samples'.format(labels.shape[0]))
    print('{} labels'.format(no_labels))
    print('{} tokens'.format(no_tokens))
    assert len(data) == labels.shape[0]
    for kfold in range(args.k_fold):
        data_seed = 42+kfold
        print('Starting {} cross validation'.format(kfold+1))
        network = get_damd_cnn(no_tokens, no_labels, args.final_nonlinearity, args.kernel_size_1, args.kernel_size_2)
        train_network_batchwise(data, labels, network, args.no_epochs, args.batch_size, args.testset_size,
                                random_state=data_seed, save_path=args.save_path, labels_to_names=args.labels_to_names,
                                loss=args.loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains CNN on indexed glogs.')
    subparsers = parser.add_subparsers(dest="command")
    # train CNN model
    train_parser = subparsers.add_parser("train_model", help="Trains model on glog data.")
    train_parser.add_argument("data_path", type=str, help="Path to indexed glogs (list of lists as .pkl file)")
    train_parser.add_argument("label_path", type=str, help="Path to labels of data (numpy array as .npy file)")
    train_parser.add_argument("fcall_path", type=str, help="Path to file containing all glog fcalls (as textfile)")
    train_parser.add_argument("--no_epochs", type=int, help="Number of epochs to train the model", default=50)
    train_parser.add_argument("--batch_size", type=int, help="Batch size at training", default=10)
    train_parser.add_argument("--testset_size", type=float, help="Fraction of data used for testing", default=0.3)
    train_parser.add_argument("--kernel_size_1", type=int, help="Kernel size of first conv layer", default=25)
    train_parser.add_argument("--kernel_size_2", type=int, help="Kernel size of seconds conv layer", default=5)
    train_parser.add_argument("--k_fold", type=int, help="Number of k-fold cross validations", default=1)
    train_parser.add_argument("--save_path", type=str, help="Where to save model architectures", default='models')
    train_parser.add_argument("--labels_to_names", type=str,
                              help="Path to file where each line is name of label dimension")
    train_parser.add_argument("--final_nonlinearity", type=str,
                              help="Nonlinearity function at final layer (one-class vs multi-class)",
                              choices=['softmax', 'sigmoid'], default='sigmoid')
    train_parser.add_argument("--loss", type=str, help="Loss function to minimize (one-class vs multi-class)",
                              choices=['categorical_crossentropy', 'binary_crossentropy'],
                              default='categorical_crossentropy')

    args = parser.parse_args()
    if args.command == "train_model":
        train_model_args(args)
