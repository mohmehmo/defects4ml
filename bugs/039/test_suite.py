import os
import pytest

def run_buggy():
    os.system('python buggy/train.py')

def run_fixed():
    os.system('python fixed/train.py')

def test_buggy():
    with pytest.raises(Exception) as e_info:
        run_buggy()
        print(e_info)

def test_fixed():
    with pytest.raises(Exception) as e_info:
        run_fixed()
        print(e_info)
