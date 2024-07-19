import pandas as pd
import hydra
from omegaconf import DictConfig
import os
from great_expectations.data_context import FileDataContext
from pathlib import Path
from hydra import compose, initialize
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import scipy.stats as stats
import great_expectations as gx
import zenml

# @hydra.main(version_base=None, config_path="../configs", config_name="config")
def sample_data():
    initialize(config_path="../configs", version_base="1.1")
    cfg = compose(config_name="config")
    data_url = cfg.data.url
    sample_size = cfg.data.sample_size
    sample_file = cfg.data.sample_file
    version = cfg.data.data_version

    # Read the data
    df = pd.read_csv(data_url)

    # Sample the data
    start = int((version - 1)*(sample_size*len(df)))
    end = int(version*(sample_size*len(df)))
    if version >= 5.0:
        end = len(df)
    sample_df = df.iloc[start:end]

    # Ensure the samples directory exists
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)

    # Save the sampled data
    sample_df.to_csv(sample_file, index=False)
    print(f"Sampled data saved to {sample_file}")
    return sample_df


def validate_initial_data():
    context = FileDataContext(project_root_dir='services')
    sample_source = context.sources.add_or_update_pandas('data_sample')
    sample_asset = sample_source.add_csv_asset(name='data_sample_asset', filepath_or_buffer='data/samples/sample.csv')
    batch_request = sample_asset.build_batch_request()
    batches = sample_asset.get_batch_list_from_batch_request(batch_request)
    
    # Create expectations suite
    context.add_or_update_expectation_suite('expectation_suite')
    
    # Create validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name='expectation_suite',
    )
    
    validator.expect_column_values_to_not_be_null(column="sellingprice",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="year",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="vin",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="mmr",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="odometer",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="body",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="condition",mostly=0.75)
    validator.expect_column_values_to_not_be_null(column="color",mostly=0.75)

    validator.expect_column_values_to_be_unique(column='vin',mostly=0.75)
    validator.expect_column_value_lengths_to_equal(column='vin',value=17,mostly=0.75)

    validator.expect_column_values_to_match_regex(column='year',regex='[0-9]{4}$')

    validator.expect_column_values_to_be_between(column='sellingprice', min_value=100, max_value=200000,mostly=0.75)

    validator.expect_column_values_to_be_between(column='mmr', min_value=50, max_value=190000,mostly=0.75)

    validator.expect_column_values_to_be_between(column='condition', min_value=0, max_value=50,mostly=0.75)





    validator.expect_column_values_to_be_of_type(
        column='sellingprice',
        type_='float64'
    )

    validator.expect_column_values_to_be_of_type(
        column='body',
        type_='str'
    )

    validator.expect_column_values_to_be_of_type(
        column='year',
        type_='int64'
    )

    validator.expect_column_values_to_be_of_type(
        column='vin',
        type_='str'
    )

    validator.expect_column_values_to_be_of_type(
        column='condition',
        type_='float64'
    )
    validator.expect_column_values_to_be_of_type(
        column='odometer',
        type_='float64'
    )

    validator.expect_column_values_to_be_of_type(
        column='mmr',
        type_='float64'
    )

    validator.expect_column_values_to_be_of_type(
        column='color',
        type_='str'
    )
    
    
    validator.save_expectation_suite(
        discard_failed_expectations = False
    )
    
    
    checkpoint = context.add_or_update_checkpoint(
        name="checkpoint",
        validator=validator,
    )
    
    
    checkpoint_result = checkpoint.run()
    print(checkpoint_result.success)
    print(checkpoint_result)
    return checkpoint_result.success


def read_datastore() -> tuple[pd.DataFrame, str]:
    initialize(config_path="../configs", version_base="1.1")
    cfg = compose(config_name="config")
    version_num = cfg.data.data_version
    print(version_num)
    sample_path = Path("/mnt/c/Users/danil/Desktop/try_2/MLOps/data") / "samples" / "sample.csv"
    df = pd.read_csv(sample_path)
    return df, version_num



def preprocess_data(df):
    df['trim'] = df['trim'].fillna('other')
    df['color'] = df['color'].fillna('other')
    df['make'] = df['make'].fillna('other')
    df['model'] = df['model'].fillna('other')


    # Fill missing values with mode
    print(df.shape)
    print('---------------------------------------------------------')
    df['body'] = df['body'].fillna(df['body'].mode()[0])
    df['interior'] = df['interior'].fillna(df['interior'].mode()[0])
    df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])

    # Remove null values
    df.dropna(subset=['vin'], inplace=True)
    df.dropna(subset=['saledate'], inplace=True)
    df['condition'] = df['condition'].fillna(df['condition'].median())
    df['odometer'] = df['odometer'].fillna(df['odometer'].mean())
    df['mmr'] = df['mmr'].fillna(df['mmr'].mean())
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    z_scores = stats.zscore(df[numerical_columns])
    clean_df = df[(z_scores < 2).all(axis=1)]
    clean_df.drop(columns=['saledate'], inplace=True)
    normalized_df = df.copy()

    # Numerical columns to be normalized
    numerical_cols = ['condition', 'odometer', 'mmr']

    # Categorical columns to be encoded
    categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'sellingprice']

    # Normalize numerical features using Min-Max Scaling
    scaler_dict = {}
    for col in numerical_cols:
        scaler = MinMaxScaler()
        normalized_df[col] = scaler.fit_transform(df[[col]])
        scaler_dict[col] = scaler

    # Encode categorical features using Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        normalized_df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X, y = normalized_df.drop(['sellingprice', 'vin'],axis=1), normalized_df[['sellingprice']]
    return X, y



def validate_features(car_prices_dataframe_tuple):
    
    context = gx.get_context()
    ds = context.sources.add_or_update_pandas(name = "transformed_data")
    da = ds.add_dataframe_asset(name = "pandas_dataframe")
    print(car_prices_dataframe_tuple[0].info())
    batch_request = da.build_batch_request(dataframe = car_prices_dataframe_tuple[0])
    context.add_or_update_expectation_suite('transformed_data_expectation')
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name='transformed_data_expectation',
    )
    validator.expect_column_values_to_not_be_null("year")
    
    validator.expect_column_values_to_be_of_type(
        column='year',
        type_='int64'
    )

    
    validator.expect_column_values_to_be_of_type(
        column='make',
        type_='int64'
    )

    
    validator.expect_column_values_to_be_of_type(
        column='model',
        type_='int64'
    )

    validator.expect_column_values_to_be_of_type(
        column='trim',
        type_='int64'
    )

    validator.expect_column_values_to_be_of_type(
        column='body',
        type_='int64'
    )
    validator.expect_column_values_to_be_of_type(
        column='transmission',
        type_='int64'
    )
    

    validator.expect_column_values_to_be_of_type(
        column='state',
        type_='int64'
    )
    validator.expect_column_values_to_be_of_type(
        column='condition',
        type_='float64'
    )
    validator.expect_column_values_to_be_of_type(
        column='odometer',
        type_='float64'
    )
    validator.expect_column_values_to_be_of_type(
        column='color',
        type_='int64'
    )
    validator.expect_column_values_to_be_of_type(
        column='interior',
        type_='int64'
    )
    validator.expect_column_values_to_be_of_type(
        column='seller',
        type_='int64'
    )
    validator.expect_column_values_to_be_of_type(
        column='mmr',
        type_='float64'
    )
    
    validator.save_expectation_suite(
        discard_failed_expectations = False
    )
    
    
    checkpoint = context.add_or_update_checkpoint(
        name="checkpoint",
        validator=validator,
    )
    
    
    checkpoint_result = checkpoint.run()


    if checkpoint_result.success:
        return car_prices_dataframe_tuple
    
def load_features(X, y, ver):
    zenml.save_artifact(data = X, name = "features", tags = [ver])
    zenml.save_artifact(data = y, name = "target", tags = [ver])


def extract_data(version, cfg):
    data_path = cfg.data.data_paths[version]
    df = pd.read_csv(data_path)
    return df, version


def init_hydra():
    hydra.initialize(config_path="conf")
    cfg = hydra.compose(config_name="config")
    return cfg


if __name__ == "__main__":
    sample_data()

