#!/usr/bin/python

import yaml


with open("configs/config.yaml") as istream:
    y = yaml.safe_load(istream)
    y['data']['data_version'] = y['data']['data_version'] + 1.0

with open("configs/config.yaml", "w") as ostream:
    yaml.dump(y, ostream, default_flow_style=False, sort_keys=False)


