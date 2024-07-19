import random

import hydra
import numpy as np
import mlflow

from model import load_features, train, log_metadata

mlflow.set_tracking_uri(uri='http://192.168.164.14:8080')


def run(args):
    cfg = args

    # Ensure reproducibility by setting random seeds
    random.seed(123)
    np.random.seed(123)

    data_version = cfg.data_version
    X_train, y_train = load_features(name="features_target", version=data_version)

    X_test, y_test = load_features(name="features_target", version=data_version)


    gs = train(X_train, y_train, cfg=cfg)

    # Ensure consistent logging (uncomment if log_metadata is defined)
    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()