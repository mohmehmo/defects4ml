import os
import json

def run_buggy():
    os.system('python buggy/imdb_fasttext.py')

def run_fixed():
    os.system('python fixed/imdb_fasttext.py')

def test_buggy():
    run_buggy()
    file = open(file="buggy/result.json", mode='r')
    result = json.load(file)
    ellapsed_time = result.get('ellapsed_time')
    assert ellapsed_time <= 70

def test_fixed():
    run_fixed()
    file = open(file="fixed/result.json", mode='r')
    result = json.load(file)
    ellapsed_time = result.get('ellapsed_time')
    assert ellapsed_time <= 70
