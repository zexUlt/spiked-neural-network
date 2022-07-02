import cppyy
import pathlib
from timeit import timeit
# from scipy.optimize import minimize
from typing import Callable, List

PATH_TO_DLL = pathlib.Path().absolute() / 'build'
LIB_NAME = 'libspiked-neural-network.so'
PROJECT_ROOT_DIR = pathlib.Path().absolute()
PROJECT_INCLUDE_DIR = PROJECT_ROOT_DIR / "include"


if __name__ == '__main__':
    lib_path = PROJECT_ROOT_DIR / PATH_TO_DLL
    cppyy.add_library_path(str(PATH_TO_DLL))
    cppyy.add_include_path(str(PROJECT_INCLUDE_DIR))
    cppyy.load_library(LIB_NAME)


    cppyy.include("Optimization/Optimization.hpp")
    cxx_opt = cppyy.gbl.cxx_sdnn.optimization
    cppyy.cppexec("std::string neuronType;")
    cppyy.gbl.neuronType = "sigmoid"
    # param_pack = [1000., 500., 1., 0.9, 1.]
    
    cppyy.cppexec("cxx_sdnn::optimization::TrainedParams params{1000., 500., 1., 0.9, 1.};")
    # trainedParams = cxx_opt.TrainedParams
    # trainedParams.a = 1000.
    # trainedParams.p = 500.
    # trainedParams.k1 = 1.
    # trainedParams.k2 = 0.9
    # trainedParams.alpha = 1.

    cppyy.cppexec("std::string trainRoot;")
    cppyy.gbl.trainRoot = "./train_data/"
    print(cxx_opt.estimate_loss(cppyy.gbl.trainRoot, cppyy.gbl.params, cppyy.gbl.neuronType))
    # print(cxx_opt.run_minimize(1000., 500., 1., 0.9, 1.,cppyy.gbl.neuronType))

