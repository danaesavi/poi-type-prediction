# Example: mm_main.py --model MM-ResNet-CONCAT
import os
import pandas as pd
import numpy as np
import argparse
import time
from numpy import savez_compressed
import random
from kerastuner.tuners import RandomSearch
from kerastuner import Objective
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import BertTokenizer
from mm_models import *
from config import *
from mm_hp import *
from load_data import DataLoader
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

"""
Examples: model_name <- --model option
- CONCAT-BERT+ResNet <- MM-Resnet-CONCAT
- CONCAT-BERT+EfficientNet <- MM-EfficientNet-CONCAT
- CONCAT-BERT+Xception <- MM-Xception-CONCAT
- Attention-BERT+Xception <- MM-Xception-ATTM
- MM-Gate <- MM-Xception-GLU
- MM-XAtt <- MM-Xception-XATT
- MM-Gated-XAtt <- MM-Xception-GLUATT
"""
models_dict = {
    "MM-GRID-CONCAT": {"train": build_ConcatClf, "hp": HyperModelConcat},
    "MM-GRID-ATTM": {"train": build_AttentionClf, "hp": HyperModelAttM},
    "MM-GRID-GLU": {"train": build_GLUClf, "hp": HyperModelGLU},
    "MM-GRID-XATT": {"train": build_XAttClf, "hp": HyperModelXAtt},
    "MM-GRID-GLUATT": {"train": build_GLUATTClf, "hp": HyperModelGLUATT},

}

print("Initializing...")
# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="../../data/",
                    help='local data directory')
parser.add_argument('--res_dir', default='./res/',
                    help='Directory for results')
parser.add_argument('--mode', default='TRAINING',
                    help='HPTUNING, TRAINING')
parser.add_argument('--model', default="MM-Resnet-CONCAT",
                    help='model to run, see mmodels')
parser.add_argument('--dataset', default="AvgAll",
                    help='AvgAll, SampleWImg')
parser.add_argument('--testing', default=0,
                    help='(1 yes, 0 no) to use a sample size 100')
parser.add_argument('--seed', default=30,
                    help='the seed to set')
opt = parser.parse_args()
print(opt)
# Load options
SEED = int(opt.seed)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
TESTING = int(opt.testing)
MODEL = opt.model
grid_name = MODEL.split("-")[1]
print("grid_name", grid_name)
# PATH
DATA = opt.data_dir
DATASET = opt.dataset
RES = opt.res_dir + DATASET + "/" + MODEL + "/SEED" + str(SEED) + "/"
mode = opt.mode

if mode in {"TRAINING", "EVAL"}:
    if MODEL in hparams and DATASET in hparams[MODEL]:
        top_dropout_rate_mm, LR, NEPOCHS = hparams[MODEL][DATASET]
        print("dropout:", top_dropout_rate_mm, "lr:", LR, "epochs:", NEPOCHS)
    else:
        sys.exit("HP not defined!")
else:
    sys.exit("mode not defined! choose TRAINING, HPTUNING")

# start_time
start_time = time.time()
print("MODEL", str(MODEL), "MODE", str(mode), "DATASET", str(DATASET))
print("MAXLEN", MAX_SEQ)
print("Loading Data")
data_loader = DataLoader(DATASET, DATA, grid_name.upper(), testing=TESTING)
splits, class_weight_dict = data_loader.load_data_splits()
train_imgs, train_txt, train_labels_enc = splits["train"]["imgs"], splits["train"]["text"], splits["train"]["lbls"]
dev_imgs, dev_txt, dev_labels_enc = splits["dev"]["imgs"], splits["dev"]["text"], splits["dev"]["lbls"]
test_imgs, test_txt, test_labels_enc = splits["test"]["imgs"], splits["test"]["text"], splits["test"]["lbls"]
print("Preprocessing text...")
# TOKENIZER
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(train_txt), return_tensors="tf", max_length=MAX_SEQ, padding=True, truncation=True)
val_encodings = tokenizer(list(dev_txt), return_tensors="tf", max_length=MAX_SEQ, padding=True, truncation=True)
test_encodings = tokenizer(list(test_txt), return_tensors="tf", max_length=MAX_SEQ, padding=True, truncation=True)
# Dataset
input_imgs = "img_input"
X_tr = dict(train_encodings)
X_tr[input_imgs] = train_imgs
X_dev = dict(val_encodings)
X_dev[input_imgs] = dev_imgs
X_test = dict(test_encodings)
X_test[input_imgs] = test_imgs
print("Preprocessing Text Done")


def train_model():
    # Model
    print("Building model:", MODEL, "\n")
    model = models_dict[MODEL.replace(grid_name, "GRID")]["train"](build_txtEncoder(), build_imgEncoder(grid_name),
                                                                       top_dropout_rate_mm)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=LR),
                  metrics=["accuracy", tf.keras.metrics.AUC()])
    print(model.summary())
    # TRAIN
    history = model.fit(x=X_tr, y=train_labels_enc,
                        class_weight=class_weight_dict,
                        batch_size=BATCH_SIZE,
                        epochs=NEPOCHS,
                        validation_data=(X_dev, dev_labels_enc),
                        verbose=1,
                        )
    print()
    h = history.history
    print(h)
    return model


def tune_hp():
    hypermodel = models_dict[MODEL.replace(grid_name, "GRID")]["hp"](grid_name=grid_name)

    tuner = RandomSearch(
        hypermodel,
        objective=Objective("val_loss", direction="min"),
        max_trials=5,
        executions_per_trial=1,
        seed=SEED,
        distribution_strategy=None,
        directory='VPOIHP',
        project_name=DATASET + "-" + MODEL)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner.search_space_summary()
    tuner.search(x=X_tr, y=train_labels_enc, epochs=NEPOCHSHP, class_weight=class_weight_dict,
                 validation_data=(X_dev, dev_labels_enc), batch_size=BATCH_SIZE, callbacks=[stop_early])
    print(tuner.results_summary())

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("The hyperparameter search is complete. The optimal dropout rate is", best_hps.get('dropout_rate'),
          "and the optimal learning rate for the optimizer is.", best_hps.get('learning_rate'))

def get_metrics(y_true_enc, y_pred_logits, split, flatten_vec=False):
    def flatten_labels(labels, preds):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = np.argmax(labels, axis=1).flatten()
        return labels_flat, pred_flat

    if flatten_vec:
        y_true, y_pred = flatten_labels(y_true_enc, y_pred_logits)
    else:
        y_true, y_pred = y_true_enc, y_pred_logits
    print(y_true.shape, y_pred.shape)
    return {
        'Model': MODEL,
        'Split': split,
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, average='micro') * 100,
        'Precision': precision_score(y_true, y_pred, average='micro') * 100,
        'Recall': recall_score(y_true, y_pred, average='micro') * 100,
        'F1 (macro)': f1_score(y_true, y_pred, average='macro') * 100,
        'Precision (macro)': precision_score(y_true, y_pred, average='macro') * 100,
        'Recall (macro)': recall_score(y_true, y_pred, average='macro') * 100
    }


def eval(model):
    print("Generate predictions...")
    predictions_test = model.predict(X_test)
    predictions_dev = model.predict(X_dev)
    print("Computing metrics...")
    test_metrics = get_metrics(test_labels_enc, predictions_test, "TEST", flatten_vec=True)
    dev_metrics = get_metrics(dev_labels_enc, predictions_dev, "DEV", flatten_vec=True)

    savez_compressed(RES + "dev_pred_{}.npz".format(MODEL.lower()), predictions_dev)
    savez_compressed(RES + "test_pred_{}.npz".format(MODEL.lower()), predictions_test)
    res = pd.DataFrame([test_metrics, dev_metrics])
    res.to_csv(RES + "metrics_{}.csv".format(MODEL.lower()), index=False)
    print("metrics saved:", RES + "metrics_{}.csv".format(MODEL.lower()))

    print("Evaluate on test data")
    results = model.evaluate(X_dev, dev_labels_enc, batch_size=BATCH_SIZE)
    print("dev loss, dev acc, dev auc:", results)
    results_test = model.evaluate(X_test, test_labels_enc, batch_size=BATCH_SIZE)
    print("test loss, test acc, test auc:", results_test)


def main():
    if mode == "TRAINING":
        model = train_model()
        if not os.path.exists(RES):
            os.makedirs(RES)
        eval(model)
        print("saving model...")
        model.save_weights(RES + MODEL.lower() + "-cktp")
        print("Done!")
        print("--- {} minutes ---".format((time.time() - start_time) / 60))
    elif mode == "HPTUNING":
        print("hypertuning...")
        tune_hp()
        print("Done!")
        print("--- {} minutes ---".format((time.time() - start_time) / 60))
    else:
        sys.exit("mode not defined! choose TRAINING, HPTUNING")

if __name__ == '__main__':
    main()
