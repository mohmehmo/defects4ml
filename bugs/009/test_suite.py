import os
import pytest

def run_buggy():
    os.system('cd buggy;python gan.py')

def run_fixed():
    os.system('cd fixed;python gan.py')

def test_buggy():
    with pytest.raises(Exception) as e_info:
        run_buggy()
        print(e_info)

def test_fixed():
    with pytest.raises(Exception) as e_info:
        run_fixed()
