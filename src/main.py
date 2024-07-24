import random

import hydra
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

from model import load_features, train, log_metadata

mlflow.set_tracking_uri(uri='http://127.0 0.1:5000')


def run(args):
    cfg = args

    # Ensure reproducibility by setting random seeds
    random.seed(123)
    np.random.seed(123)

    data_version = cfg.data_version
    X, y = load_features(dataset_name="final_features_target", dataset_version=1)
    X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()