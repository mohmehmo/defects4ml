import os
import pytest

def run_buggy():
    os.system('python buggy/cnp/train.py')

def run_fixed():
    os.system('python fixed/cnp/train.py')

def test_buggy():
    with pytest.raises(Exception) as e_info:
        run_buggy()

def test_fixed():
    with pytest.raises(Exception) as e_info:
        run_fixed()