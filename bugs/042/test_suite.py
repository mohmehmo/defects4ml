import os
import json

def run_buggy():
    os.system('python buggy/mnist_odenet.py')

def run_fixed():
    os.system('python fixed/mnist_odenet.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    accuracy = result.get('accuracy')
    assert accuracy >= 0.88

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    accuracy = result.get('accuracy')
    assert accuracy >= 0.88
