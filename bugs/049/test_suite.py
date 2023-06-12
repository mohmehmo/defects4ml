import os
import json

def run_buggy():
    os.system('cd buggy;python gru_jena_climate.py')

def run_fixed():
    os.system('cd fixed;python gru_jena_climate.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    assert loss <= 0.05

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    loss = result.get('loss')
    assert loss <= 0.05
