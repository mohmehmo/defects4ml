import os
import pytest

def run_buggy():
    os.system('python buggy/keras_cnn_example.py')

def run_fixed():
    os.system('python fixed/keras_cnn_example.py')

def test_buggy():
    with pytest.raises(Exception) as e_info:
        run_buggy()
        print(e_info)

def test_fixed():
    with pytest.raises(Exception) as e_info:
        run_fixed()
