#!/usr/bin/python

import yaml


with open("/mnt/c/Users/danil/Desktop/try_2/MLOps/configs/config.yaml") as istream:
    y = yaml.safe_load(istream)
    y['data']['data_version'] = y['data']['data_version'] + 1.0

with open("/mnt/c/Users/danil/Desktop/try_2/MLOps/configs/config.yaml", "w") as ostream:
    yaml.dump(y, ostream, default_flow_style=False, sort_keys=False)


with open("/mnt/c/Users/danil/Desktop/try_2/MLOps/configs/data_version.yaml") as istream:
    y = yaml.safe_load(istream)
    y['data_version'] = y['data_version'] + 1.0

with open("/mnt/c/Users/danil/Desktop/try_2/MLOps/configs/data_version.yaml", "w") as ostream:
    yaml.dump(y, ostream, default_flow_style=False, sort_keys=False)


