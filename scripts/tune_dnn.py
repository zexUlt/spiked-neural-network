import cppyy
import pathlib
from timeit import timeit
from scipy.optimize import minimize
from typing import Callable, List

PATH_TO_DLL = pathlib.Path().absolute() / 'build'
LIB_NAME = 'libsdnn.so'
PROJECT_ROOT_DIR = pathlib.Path().absolute()
PROJECT_INCLUDE_DIR = PROJECT_ROOT_DIR / "include"


def wrapper(x, *args):
    vec = cppyy.gbl.std.vector[float](x)
    return cxx_opt.run_minimize(vec, *args)

def what_is_happening(x):
    print(f"Current state vector is {x}\n")


lib_path = PROJECT_ROOT_DIR / PATH_TO_DLL
cppyy.add_library_path(str(PATH_TO_DLL))
cppyy.add_include_path(str(PROJECT_INCLUDE_DIR))
cppyy.load_library(LIB_NAME)


cppyy.include("Optimization/Optimization.hpp")
cxx_opt = cppyy.gbl.cxx_sdnn.optimization
cppyy.cppexec("std::string neuronType;")
cppyy.gbl.neuronType = "sigmoid"
param_pack = [162., 3337., 1., 0.1, 1., 1.0375, 0.975, 0.01925, -0.02075, 0.]


cppyy.cppexec("std::string trainRoot;")
cppyy.gbl.trainRoot = "./train_data/"
# print(cxx_opt.estimate_loss(cppyy.gbl.trainRoot, cppyy.gbl.params, cppyy.gbl.neuronType))
x_opt = minimize(wrapper, param_pack, 
    args=(cppyy.gbl.neuronType, cppyy.gbl.trainRoot), 
    method='Nelder-Mead', callback=what_is_happening,
    options=dict(disp=True))
print(x_opt)

