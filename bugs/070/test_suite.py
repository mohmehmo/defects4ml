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
    loss = result.get('loss')
    assert loss <= 50

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    assert loss <= 50
