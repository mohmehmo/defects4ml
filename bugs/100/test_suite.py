import os
import json

def run_buggy():
    os.system('cd buggy;python script.py')

def run_fixed():
    os.system('cd fixed;python script.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    print(loss)
    assert loss != "nan"

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    print(loss)
    assert loss != "nan"