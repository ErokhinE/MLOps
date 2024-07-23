#!/usr/bin/python

import yaml
import os

PROJECT_PATH = os.environ['PROJECT_DIR']
with open(f"{PROJECT_PATH}/configs/config.yaml") as istream:
    y = yaml.safe_load(istream)
    y['data']['data_version'] = y['data']['data_version'] + 1.0

with open(f"{PROJECT_PATH}/configs/config.yaml", "w") as ostream:
    yaml.dump(y, ostream, default_flow_style=False, sort_keys=False)


with open(f"{PROJECT_PATH}/configs/data_version.yaml") as istream:
    y = yaml.safe_load(istream)
    y['data_version'] = y['data_version'] + 1.0

with open(f"{PROJECT_PATH}/configs/data_version.yaml", "w") as ostream:
    yaml.dump(y, ostream, default_flow_style=False, sort_keys=False)


