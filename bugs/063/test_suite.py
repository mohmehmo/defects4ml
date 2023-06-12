import os
import json

def run_buggy():
    os.system('cd buggy/LeNet/Keras;python main.py')

def run_fixed():
    os.system('cd fixed/LeNet/Keras;python main.py')

def test_buggy():
    avg_acc = 0
    for i in range(10):
        run_buggy()
        file = open(file="buggy/LeNet/Keras/result.json", mode='r')
        result = json.load(file)
        acc = result.get('val_acc')
        avg_acc = avg_acc + acc
    assert avg_acc/10 >= 0.465

def test_fixed():
    avg_acc = 0
    for i in range(10):
        run_fixed()
        file = open(file="fixed/LeNet/Keras/result.json", mode='r')
        result = json.load(file)
        acc = result.get('val_acc')
        avg_acc = avg_acc + acc
    assert avg_acc >= 0.465
