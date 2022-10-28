from typing import Callable
import numpy as np
from main import activation_function

def sigm(x):
    activation_function(x, [])

def prepare_data():
    raw_data = np.load('../train_data/tr_target.npy')
    target_coord = raw_data[1:]
    target_speed = np.diff(raw_data, 1, axis=0) * 120.
    target_data = np.concatenate((target_coord, target_speed))
    
    return target_data


if __name__ == '__main__':
    input_data = prepare_data()
    
    assert(sigm(input_data), np.load('cxx_sigm.npy'))    