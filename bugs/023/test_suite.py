import os
import json

def run_buggy():
    os.system('python buggy/train.py --input images/ --size 250 250')

def run_fixed():
    os.system('python fixed/train.py --input images/ --size 250 250')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    assert loss <= 0.1

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    assert loss <= 0.1