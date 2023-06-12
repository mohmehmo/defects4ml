import os
import json

def run_buggy():
    os.system('python buggy/mnist_resnet.py')

def run_fixed():
    os.system('python fixed/mnist_resnet.py')

def test_buggy():
    avg_acc = 0
    for i in range(10):
        run_buggy()
        file = open(file="buggy/result.json", mode='r')
        result = json.load(file)
        accuracy = result.get('accuracy')
        avg_acc = avg_acc + accuracy
    assert avg_acc/10 >= 0.992

def test_fixed():
    avg_acc = 0
    for i in range(10):
        run_buggy()
        file = open(file="fixed/result.json", mode='r')
        result = json.load(file)
        accuracy = result.get('accuracy')
        avg_acc = avg_acc + accuracy
    assert avg_acc/10 >= 0.992
