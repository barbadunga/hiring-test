import logging
import argparse
from pathlib import Path
import json
import multiprocessing as mp
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

# https://github.com/SergeyShk/Word-to-Number-Russian/blob/master/number.py


def get_data(args):
    df_train = pd.read_csv(args.train)
    if args.test is None:
        train, test = train_test_split(df_train, test_size=.2)
    else:
        test = pd.read_csv(args.test)
        train = df_train
    return train, test


def multiprocess_text(model, data, workers=5):
    chunks = np.array_split(data, workers)
    LOGGER.info("Start text processing")
    start_time = time.time()
    with mp.Pool(workers) as p:
        res = pd.concat((p.map(model.fit_transform, chunks)))
    end_time = time.time()
    LOGGER.info("Finish text processing in %.3f" % (end_time - start_time))
    return res


def get_json(data_path):
    with open(data_path, "r") as file:
        data = json.load(file)
    return data


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default="/hiring-test-data/train.csv")
    parser.add_argument("--test", type=Path, default="/hiring-test-data/test.csv")
    parser.add_argument("-o", "--output", type=Path, default="/hiring-test-data/prediction.csv")
    return parser.parse_args()


def cv_roc_auc(clf, X, y, sample_weight=None):
    """ Calculate roc-auc score for CV """
    y_pred = clf.predict_proba(X)
    return roc_auc_score(y, y_pred[:, 1])


def auc_by_category(test, y_pred):
    """ Calculate roc_auc for each unique category in dataset """
    result = dict()
    for cat in test["category"].unique():
        mask = test["category"] == cat
        LOGGER.info("Estimate %s with shape %d" % (cat, test[mask].shape[0]))
        score = roc_auc_score(test.loc[mask, "is_bad"], y_pred[mask])
        result[cat] = score
        LOGGER.info("ROC-AUC: %.5f" % score)
    return result
