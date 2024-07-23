import importlib
import random
import mlflow
import numpy as np
import pandas as pd
from zenml.client import Client
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import mlflow.pyfunc
from zenml.artifacts.utils import load_artifact
from joblib import Parallel, parallel_backend
import matplotlib
# import shap
import logging

logging.basicConfig(level=logging.DEBUG)
shap_logger = logging.getLogger('shap')
shap_logger.setLevel(logging.DEBUG)

mlflow.set_tracking_uri(uri='http://192.168.164.14:5000')

def load_features(dataset_name, dataset_version, sample_fraction=1):
    dataframe = load_artifact(name_or_id='final_features_target', version=dataset_version)
    sampled_dataframe = dataframe.sample(
        frac=sample_fraction, random_state=88
    )
    feature_data = sampled_dataframe[sampled_dataframe.columns[:-1]]
    target_data = sampled_dataframe[sampled_dataframe.columns[-1]]
    return feature_data, target_data

def get_model(model_name):
    if model_name == "random_forest_regressor":
        return RandomForestRegressor()
    elif model_name == "gradient_boosting_regressor":
        return GradientBoostingRegressor()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train(X_train, y_train, cfg):
    random.seed(123)
    np.random.seed(123)
    params = cfg.model.params
    print(params)
    model = get_model(cfg.model.model_name)
    param_grid = dict(params)
    scoring = list(cfg.model.metrics.values())
    evaluation_metric = cfg.model.evaluation_metric
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=1,
        refit=evaluation_metric,
        cv=cfg.model.folds,
        verbose=2,
        return_train_score=True,
    )
    with parallel_backend('loky', n_jobs=-1):
        gs.fit(X_train, y_train)
    return gs

def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    # cross_val_results = (
    #     pd.DataFrame(grid_search.cv_results_)
    #     .filter(regex=r"std_|mean_|param_")
    #     .sort_index(axis=1)
    # )
    # optimal_metrics_values = [
    #     result[1][grid_search.best_index_] for result in grid_search.cv_results_.items()
    # ]
    # optimal_metrics_keys = [metric for metric in grid_search.cv_results_]
    # optimal_metrics_dict = {
    #     key: value
    #     for key, value in zip(optimal_metrics_keys, optimal_metrics_values)
    #     if "mean" in key or "std" in key
    # }

    # hyperparameters = optimal_metrics_dict

    # train_dataframe = pd.concat([train_features, train_target], axis=1)
    # test_dataframe = pd.concat([test_features, test_target], axis=1)

    # experiment_title = config.model_name + "_experiment_model"

    # try:
    #     experiment_id = mlflow.create_experiment(name=experiment_title)
    # except mlflow.exceptions.MlflowException:
    #     experiment_id = mlflow.get_experiment_by_name(name=experiment_title).experiment_id

    # print("Experiment ID: ", experiment_id)

    # cv_metric = config.cv_evaluation_metric
    # run_title = "_".join(['model_run', config.model_name, config.evaluation_metric, str(hyperparameters["mean_test_"+cv_metric]).replace(".", "_")])
    # print("Run title: ", run_title)

    # if mlflow.active_run():
    #     mlflow.end_run()

    # with mlflow.start_run():
    #     pass

    # with mlflow.start_run(run_name=run_title, experiment_id=experiment_id) as parent_run:
    #     train_dataset = mlflow.data.pandas_dataset.from_pandas(df=train_dataframe, targets='sellingprice')
    #     test_dataset = mlflow.data.pandas_dataset.from_pandas(df=test_dataframe, targets='sellingprice')
    #     mlflow.log_input(train_dataset, "training")
    #     mlflow.log_input(test_dataset, "testing")

    #     mlflow.log_params(grid_search.best_params_)
    #     mlflow.log_metrics(optimal_metrics_dict)

    #     mlflow.set_tag('regressor', config.model_name)
    #     model_signature = mlflow.models.infer_signature(train_features, grid_search.predict(train_features))

    #     # Register the model
    #     model_info = mlflow.sklearn.log_model(
    #         sk_model=grid_search.best_estimator_,
    #         artifact_path=config.model_name + '_model',
    #         signature=model_signature,
    #         input_example=train_features,
    #         registered_model_name=config.model_name,
    #     )

    #     # Get the latest version of the model
    #     client = mlflow.client.MlflowClient()
    #     model_versions = client.get_latest_versions(config.model_name, stages=["None"])
    #     model_version = max([int(version.version) for version in model_versions])

    #     client.set_model_version_tag(
    #         name=config.model_name,
    #         version=model_version,
    #         key="source",
    #         value="best_grid_search_model",
    #     )

    #     test_predictions = grid_search.best_estimator_.predict(test_features)
    #     evaluation_data = pd.DataFrame(test_target)
    #     evaluation_data.columns = ["label"]
    #     evaluation_data["predictions"] = test_predictions

    #     evaluation_results = mlflow.evaluate(
    #         data=evaluation_data,
    #         model_type="regressor",
    #         targets="label",
    #         predictions="predictions",
    #         evaluators=["default"],
    #     )

    #     mlflow.log_metrics(evaluation_results.metrics)

    #     print(f"Best model metrics:\n{evaluation_results.metrics}")

    #     for idx, result in cross_val_results.iterrows():
    #         child_run_title = "_".join(["child", run_title, str(idx)])

    #         with mlflow.start_run(run_name=child_run_title, experiment_id=experiment_id, nested=True):
    #             child_params = result.filter(regex="param_").to_dict()
    #             child_means = result.filter(regex="mean_").to_dict()
    #             child_stds = result.filter(regex="std_").to_dict()
    #             child_params = {key.replace("param_", ""): value for key, value in child_params.items()}
    #             mlflow.log_params(child_params)
    #             mlflow.log_metrics(child_means)
    #             mlflow.log_metrics(child_stds)
    # mlflow.end_run()
    cv_results = pd.DataFrame(gs.cv_results_).filter(regex=r'std_|mean_|param_').sort_index(axis=1)
    best_metrics_values = [result[1][gs.best_index_] for result in gs.cv_results_.items()]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {k:v for k,v in zip(best_metrics_keys, best_metrics_values) if 'mean' in k or 'std' in k}

    # print(cv_results, cv_results.columns)

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis = 1)
    df_test = pd.concat([X_test, y_test], axis = 1)

    experiment_name = cfg.model.model_name + "_test_model"
    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id # type: ignore
    
    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join(['model_run', cfg.model.model_name, cfg.model.evaluation_metric, str(params['mean_test_'+cv_evaluation_metric]).replace(".", "_")]) # type: ignore
    print("run name: ", run_name)

    if (mlflow.active_run()):
        mlflow.end_run()

    # Fake run
    with mlflow.start_run():
        pass

    # Parent run
    with mlflow.start_run(run_name = run_name, experiment_id = experiment_id) as run:

        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_train, targets = 'sellingprice') # type: ignore
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_test, targets = 'sellingprice') # type: ignore
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model = gs.best_estimator_,
            artifact_path = cfg.model.artifact_path,
            signature = signature,
            input_example = X_train.iloc[0].to_numpy(),
            registered_model_name = cfg.model.model_name,
            pyfunc_predict_fn = cfg.model.pyfunc_predict_fn
        )
        client = mlflow.client.MlflowClient()
        try:
            client.set_model_version_tag(name = cfg.model.model_name, version=model_info.registered_model_version, key="source", value="best_Grid_search_model")
        except e:
            pass
        for index, result in cv_results.iterrows():
            child_run_name = "_".join(['child', run_name, str(index)]) # type: ignore
            with mlflow.start_run(run_name = child_run_name, experiment_id=experiment_id, nested=True) as second_run: #, tags=best_metrics_dict):
                ps = result.filter(regex='param_').to_dict()
                ms = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()
                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_",""):v for (k,v) in ps.items()}
                # if 'max_depth' in ps.keys():
                #     ps['max_depth'] = int(ps['max_depth'])

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name # e.g. "sklearn.linear_model"
                class_name  = cfg.model.class_name # e.g. "LogisticRegression"

                # Load "module.submodule.MyClass"
                class_instance = getattr(importlib.import_module(module_name), class_name)
                
                estimator = class_instance(**ps)
                estimator.fit(X_train, y_train)
                
                signature = mlflow.models.infer_signature(X_train, estimator.predict(X_train))

                model_info = mlflow.sklearn.log_model(
                    sk_model = estimator,
                    artifact_path = cfg.model.artifact_path,
                    signature = signature,
                    input_example = X_train.iloc[0].to_numpy(),
                    registered_model_name = cfg.model.model_name,
                    pyfunc_predict_fn = cfg.model.pyfunc_predict_fn
                )

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)
                predictions = loaded_model.predict(X_test) # type: ignore
                
                eval_data = pd.DataFrame(y_test)
                eval_data.columns = ["sellingprice"]
                eval_data["predictions"] = predictions

                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="regressor",
                    targets="sellingprice",
                    predictions="predictions",
                    evaluators=["default"]
                )
                mlflow.log_metrics(results.metrics)

               
            


def retrieve_model_with_alias(model_name, model_alias):
    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
