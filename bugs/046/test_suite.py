import os
import json

def run_buggy():
    os.system('python buggy/XOR.py')

def run_fixed():
    os.system('python fixed/XOR.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    accuracy = result.get('prediction')
    assert accuracy == [[0], [1.0], [1.0], [0]]

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    accuracy = result.get('prediction')
    assert accuracy == [[0], [1.0], [1.0], [0]]
