# coding=utf-8
import json


def data_parse(file):
    with open(file, 'r') as fp:
        config = json.loads(fp.read())
        pass
    return config
    pass


def train_parse(file):
    with open(file, 'r') as fp:
        config = json.loads(fp.read())
        pass
    return config
    pass


pass
