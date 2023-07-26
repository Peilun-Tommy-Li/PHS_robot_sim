# PHS_simulation
# Author: Tommy Li
# Date: Jul.10, 2023
# Description: The Python interface for the kernel function. This is the interface that provides
#               abstraction to the C code. This function is meant to be imported by other files
#               that need to call the covariance kernel. The C code is compile to the shared lib
#               file format(.so), and this function needs "PHSKernel_se.so" file in the directory.

import ctypes
import numpy as np


def PHS_kernel(A, B, hyp_sd, hyp_l, d1):
    # Load the shared library
    mylib = ctypes.CDLL("./PHSKernel_se.so")

    # Define the argument types for the C function
    mylib.kernel.argtypes = (
        ctypes.POINTER(ctypes.c_double),  # A
        ctypes.POINTER(ctypes.c_double),  # B
        ctypes.c_double,  # hyp_sd
        ctypes.POINTER(ctypes.c_double),  # hyp_l
        ctypes.c_int,  # d1
        ctypes.POINTER(ctypes.c_double),  # result
        ctypes.c_int,  # rowA
        ctypes.c_int,  # colA
        ctypes.c_int,  # rowB
        ctypes.c_int,  # colB
        ctypes.c_int  # rowC
    )
    rowA = A.shape[0]
    colA = A.shape[1]
    rowB = B.shape[0]
    colB = B.shape[1]
    rowC = len(hyp_l)

    # Define the return type
    mylib.kernel.restype = None

    kernel = mylib.kernel

    # type conversion to C
    # Convert input matrices to appropriate types
    A_ptr = np.ascontiguousarray(np.array(A).T, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    B_ptr = np.ascontiguousarray(np.array(B).T, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Convert hyp_l to appropriate type
    hyp_l_ptr = np.ascontiguousarray(np.array(hyp_l), dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    print('.', end='')
    if d1 == 2:
        # Create the result matrix
        dims = (colA * rowA, colB * rowA)
        outMatrix = np.zeros(dims, dtype=np.float64)

        # convert C to C_ptr
        C_ptr = np.ascontiguousarray(outMatrix, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call the C function with the result matrix
        kernel(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, C_ptr, rowA, colA, rowB, colB, rowC)
    elif d1 == 1:
        # Create the result matrix
        dims = (colB, colA * rowA)
        outMatrix = np.empty(dims, dtype=np.float64)

        # convert C to C_ptr
        C_ptr = np.ascontiguousarray(outMatrix, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call the C function with the result matrix
        kernel(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, C_ptr, rowA, colA, rowB, colB, rowC)
    else:
        # Create the result matrix
        dims = (colB, colA)
        outMatrix = np.empty(dims, dtype=np.float64)

        # convert C to C_ptr
        C_ptr = np.ascontiguousarray(outMatrix, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        kernel(A_ptr, B_ptr, hyp_sd, hyp_l_ptr, d1, C_ptr, rowA, colA, rowB, colB, rowC)

    return outMatrix.T
