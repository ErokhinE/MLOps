import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
import os
import sys
PROJECT_PATH = os.environ['PROJECT_DIR']
sys.path.append(os.path.abspath(PROJECT_PATH))
from src.data import read_datastore, preprocess_data, validate_features, load_features

os.chdir(PROJECT_PATH)


@step(enable_cache=False)
def data_extraction() -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="extracted_data", tags=["data_pipeline"])
    ],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_pipeline"])],
]:
    df, version = read_datastore()
    return df, str(version)

@step(enable_cache=False)
def data_transformation(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="transformed_features", tags=["data_pipeline"])
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="transformed_target", tags=["data_pipeline"])
    ],
]:
    # Your data transformation code
    X, y = preprocess_data(df)
    return X, y

@step(enable_cache=False)
def data_validation(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="validated_features", tags=["data_pipeline"]),
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="validated_target", tags=["data_pipeline"])
    ],
]:
    X, y = validate_features((X, y))
    return X, y

@step(enable_cache=False)
def data_loading(
    X: pd.DataFrame, y: pd.DataFrame, version: str
) -> Annotated[
    pd.DataFrame, ArtifactConfig(name="final_features_target", tags=["data_pipeline"])
]:
    load_features(X, y, version)
    return pd.concat([X, y], axis=1)

@pipeline()
def data_pipeline():
    df, version = data_extraction()
    X, y = data_transformation(df)
    X, y = data_validation(X, y)
    df = data_loading(X, y, version)

if __name__ == "__main__":
    data_pipeline()