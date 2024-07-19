from data import extract_data # custom module
from transform_data import transform_data # custom module
from model import retrieve_model_with_alias # custom module
from utils import init_hydra
import giskard
import hydra
import mlflow

def prepare_dataset(cfg):
    version = cfg.data.data_version
    df, version = extract_data(version=version, cfg=cfg)
    target = cfg.data.target
    cat_cols = list(cfg.data.target)
    data_name = f'{cfg.data.dataset_name}.{cfg.data.version}'
    giskard_dataset = giskard.Dataset(
        df=df,
        target=target,
        name=data_name,
        cat_columns=cat_cols
    )
    return giskard_dataset, df, version


def load_model(model_name, model_alias):
    client = mlflow.MlflowClient()
    model = retrieve_model_with_alias(model_name, model_alias=model_alias)
    model_version = client.get_model_version_by_alias(name=model_name, alias=model_alias).version
    return model, model_version


def predict(cfg, df, model, version, transformer_version):
    X = transform_data(
        df=df,
        version=version,
        cfg=cfg,
        return_df=False,
        only_transform=True,
        transformer_version=transformer_version,
        only_X=True
    )
    return model.predict(X)

def create_giskard_model(predict_func, model_name, cfg, df):
    giskard_model = giskard.Model(
        model=predict_func,
        model_type="regression",
        classification_labels=list(cfg.data.labels),
        feature_names=df.columns,
        name=model_name
    )
    return giskard_model

def run_giskard_scan(giskard_model, giskard_dataset, model_name, model_version, dataset_name, version):
    scan_results = giskard.scan(giskard_model, giskard_dataset)
    scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
    scan_results.to_html(scan_results_path)
    return scan_results_path

def create_and_run_test_suite(giskard_model, giskard_dataset, model_name, model_version, dataset_name, version, threshold):
    suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
    test_suite = giskard.Suite(name=suite_name)

    test1 = giskard.testing.test_f1(model=giskard_model, dataset=giskard_dataset, threshold=threshold)
    test_suite.add_test(test1)
    
    test_results = test_suite.run()
    return test_results

def select_best_model(cfg, giskard_dataset, df, version):
    model_names = cfg.model.challenger_model_names
    model_aliases = ["challenger" + str(i+1) for i in range(len(model_names))]
    evaluation_metric_threshold = cfg.model.f1_threshold
    transformer_version = cfg.data_transformer_version

    client = mlflow.MlflowClient()
    best_model = None
    least_issues = float('inf')

    for model_name, model_alias in zip(model_names, model_aliases):
        model, model_version = load_model(model_name, model_alias)
        predict_func = predict(cfg, df, model, version, transformer_version)
        giskard_model = create_giskard_model(predict_func, model_name, cfg, df)
        run_giskard_scan(giskard_model, giskard_dataset, model_name, model_version, giskard_dataset.name, version)
        test_results = create_and_run_test_suite(giskard_model, giskard_dataset, model_name, model_version, giskard_dataset.name, version, evaluation_metric_threshold)

        if test_results.passed:
            num_issues = len(test_results.results) - sum([1 for result in test_results.results if result.passed])
            if num_issues < least_issues:
                least_issues = num_issues
                best_model = (model_name, model_version, model_alias)

    return best_model

def tag_and_deploy_best_model(best_model):
    if best_model:
        model_name, model_version, model_alias = best_model
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        print(f"Model {model_name} version {model_version} is tagged as the champion and deployed.")
    else:
        print("No model found.")

def main():
    cfg = init_hydra()
    giskard_dataset, df, version = prepare_dataset(cfg)
    best = select_best_model(cfg, giskard_dataset, df, version)
    tag_and_deploy_best_model(best)

if __name__ == '__main__':
    main()