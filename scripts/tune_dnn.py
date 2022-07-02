import cppyy
import pathlib
from timeit import timeit
# from scipy.optimize import minimize
from typing import Callable, List

PATH_TO_DLL = pathlib.Path().absolute() / 'build'
LIB_NAME = 'libspiked-neural-network.so'
PROJECT_ROOT_DIR = pathlib.Path().absolute()
PROJECT_INCLUDE_DIR = PROJECT_ROOT_DIR / "include"

def run_dnn(func: Callable, x0: List[float], act_type: str) -> float:
    return func(*x0, act_type)

if __name__ == '__main__':
    lib_path = PROJECT_ROOT_DIR / PATH_TO_DLL
    
    # cppyy.include(PROJECT_INCLUDE_DIR / "Optimization/Optimization.hpp")
    cppyy.add_library_path(str(PATH_TO_DLL))
    cppyy.add_include_path(str(PROJECT_INCLUDE_DIR))
    cppyy.load_library(LIB_NAME)
    cppyy.include("Optimization/Optimization.hpp")
    # act_type = "sigmoid".encode("utf-8")
    # x0 = [1000, 500, 1, 0.9]

    # ret = run_dnn(functional, x0, act_type)

    # print(ret)
