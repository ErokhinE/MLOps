import pandas as pd
import hydra
from omegaconf import DictConfig
import os
from great_expectations.data_context import FileDataContext

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def sample_data(cfg: DictConfig):
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
    
    validator.expect_column_values_to_not_be_null(column="sellingprice",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="year",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="vin",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="mmr",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="odometer",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="body",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="condition",mostly=0.95)
    validator.expect_column_values_to_not_be_null(column="color",mostly=0.95)

    validator.expect_column_values_to_be_unique(column='vin',mostly=0.95)
    validator.expect_column_value_lengths_to_equal(column='vin',value=17,mostly=0.95)

    validator.expect_column_values_to_match_regex(column='year',regex='[0-9]{4}$')

    validator.expect_column_values_to_be_between(column='sellingprice', min_value=100, max_value=200000,mostly=0.95)

    validator.expect_column_values_to_be_between(column='mmr', min_value=50, max_value=190000,mostly=0.95)

    validator.expect_column_values_to_be_between(column='condition', min_value=0, max_value=50,mostly=0.95)





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
    
    # Store expectation suite
    validator.save_expectation_suite(
        discard_failed_expectations = False
    )
    
    # Create checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name="checkpoint",
        validator=validator,
    )
    
    # Run validation
    checkpoint_result = checkpoint.run()
    return checkpoint_result.success




if __name__ == "__main__":
    sample_data()
    validate_initial_data()
