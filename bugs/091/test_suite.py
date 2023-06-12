import os
import json

def run_buggy():
    os.system('python buggy/script.py')

def run_fixed():
    os.system('python fixed/script.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    acc = result.get('val_acc')
    assert acc >= 0.85

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    acc = result.get('val_acc')
    assert acc >= 0.85
