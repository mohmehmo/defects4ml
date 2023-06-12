import os
import json

def run_buggy():
    os.system('python buggy/train.py')

def run_fixed():
    os.system('python fixed/train.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    acc = result.get('test_accuracy')
    assert acc >= 0.99

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    acc = result.get('test_accuracy')
    assert acc >= 0.99
