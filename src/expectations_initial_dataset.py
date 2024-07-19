import great_expectations as ge
from great_expectations.dataset import PandasDataset
import pandas as pd

car_prices = pd.read_csv('data/car_prices.csv')
ge_car_prices = ge.from_pandas(car_prices)


class DataValidationException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

ge_car_prices.expect_column_values_to_not_be_null(column="sellingprice",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="year",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="vin",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="mmr",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="odometer",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="body",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="condition",mostly=0.95)
ge_car_prices.expect_column_values_to_not_be_null(column="color",mostly=0.95)

ge_car_prices.expect_column_values_to_be_unique(column='vin',mostly=0.95)
ge_car_prices.expect_column_value_lengths_to_equal(column='vin',value=17,mostly=0.95)

ge_car_prices.expect_column_values_to_match_regex(column='year',regex='[0-9]{4}$')

ge_car_prices.expect_column_values_to_be_between(column='sellingprice', min_value=100, max_value=200000,mostly=0.95)

ge_car_prices.expect_column_values_to_be_between(column='mmr', min_value=50, max_value=190000,mostly=0.95)

ge_car_prices.expect_column_values_to_be_between(column='condition', min_value=0, max_value=50,mostly=0.95)





ge_car_prices.expect_column_values_to_be_of_type(
    column='sellingprice',
    type_='float64'
)

ge_car_prices.expect_column_values_to_be_of_type(
    column='body',
    type_='str'
)

ge_car_prices.expect_column_values_to_be_of_type(
    column='year',
    type_='int64'
)

ge_car_prices.expect_column_values_to_be_of_type(
    column='vin',
    type_='str'
)

ge_car_prices.expect_column_values_to_be_of_type(
    column='condition',
    type_='float64'
)
ge_car_prices.expect_column_values_to_be_of_type(
    column='odometer',
    type_='float64'
)

ge_car_prices.expect_column_values_to_be_of_type(
    column='mmr',
    type_='float64'
)

ge_car_prices.expect_column_values_to_be_of_type(
    column='color',
    type_='str'
)

results = ge_car_prices.validate()

# Check if the data is valid
if results['success']:
    print("All expectations passed!")
else:
    for result in results['results']:
        if not result['success']:
          print(result['expectation_config']['kwargs']['column'],':')
          print(f"Expectation: {result['expectation_config']['expectation_type']}")
          print(f"Details: {result['result']}")
          raise DataValidationException(f"Expectation: {result['expectation_config']['expectation_type']}")

