import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from code_for_learning.utils import auc_by_category, parse_args, get_data, get_json, multiprocess_text
from code_for_learning.text_processing import TextNormalizer
from code_for_learning.transformer import Vectorizer

logging.basicConfig(format='%(asctime)-15s | %(levelname)s | %(name)s | %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DATA_PATH = Path("/src/data")


def save_prediction(y_pred, path):
    prediction = pd.DataFrame()
    prediction['index'] = range(y_pred.shape[0])
    prediction['prediction'] = y_pred
    prediction.to_csv(path, index=False)


def main():
    args = parse_args()

    # load all data
    train, test = get_data(args)
    first_names = get_json(DATA_PATH / "valid_names.json")
    digits = get_json(DATA_PATH / "digits.json")

    # prepare text features
    text_normalize = TextNormalizer(names=first_names, digits=digits)
    LOGGER.info("Processing train text ...")
    # train = multiprocess_text(text_normalize, train, workers=3)
    train = text_normalize.fit_transform(train)
    LOGGER.info("Processing test text ...")
    # test = multiprocess_text(text_normalize, test, workers=3)
    test = text_normalize.fit_transform(test)
    import gc

    gc.collect()
    # feature extraction
    LOGGER.info("Generating training dataset ...")
    vectorizer = Vectorizer(
        text_col="text",
        cat_cols=["subcategory", "region"]
    )
    X_train = vectorizer.fit_transform(train)
    LOGGER.info("Get data with shape %dx%d" % (X_train.shape[0], X_train.shape[1]))

    LOGGER.info("Generating test dataset ...")
    X_test = vectorizer.transform(test)
    LOGGER.info("Get data with shape %dx%d" % (X_test.shape[0], X_test.shape[1]))

    # train model
    model = LogisticRegression(C=1.0, penalty="l1", n_jobs=-1, solver="liblinear", random_state=42, verbose=1)
    LOGGER.info("Start training")
    model = model.fit(X_train, train["is_bad"])

    # evaluate and save result
    y_pred = model.predict_proba(X_test)
    result = auc_by_category(test, y_pred[:, 1])
    LOGGER.info("Average roc_auc by categories %.5f" % (sum(result.values()) / len(result)))
    save_prediction(y_pred[:, 1], args.output)
    LOGGER.info("Saved prediction to %s" % args.output)


if __name__ == '__main__':
    exit_code = 0
    try:
        main()
    except Exception as err:
        LOGGER.error(err.with_traceback())
        exit_code = 1
    sys.exit(exit_code)
