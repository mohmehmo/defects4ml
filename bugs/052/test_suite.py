import os
import json

def run_buggy():
    os.system('python buggy/fashion_mnist_cnn1_aug.py')

def run_fixed():
    os.system('python fixed/fashion_mnist_cnn1_aug.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    acc = result.get('test_acc')
    assert acc >= 0.9

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    acc = result.get('test_acc')
    assert acc >= 0.9
