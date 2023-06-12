import os
import json

def run_buggy():
    os.system('cd buggy;python simple_cnn.py')

def run_fixed():
    os.system('cd fixed;python simple_cnn.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    accuracy = result.get('accuracy')
    assert accuracy >= 0.92

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    accuracy = result.get('accuracy')
    assert accuracy >= 0.92
