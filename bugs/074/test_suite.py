import os
import json

def run_buggy():
    os.system('cd buggy;python gol.py 20 30')

def run_fixed():
    os.system('cd fixed;python gol.py 20 30')

def test_buggy():
    avg_loss = 0
    for i in range(10):
        run_buggy()
        file = open(file="buggy/result.json", mode='r')
        result = json.load(file)
        loss = result.get('loss')
        avg_loss = avg_loss + loss
    assert avg_loss <= 0.003

def test_fixed():
    avg_loss = 0
    for i in range(10):
        run_fixed()
        file = open(file="fixed/result.json", mode='r')
        result = json.load(file)
        loss = result.get('loss')
        avg_loss = avg_loss + loss
    assert avg_loss <= 0.003
