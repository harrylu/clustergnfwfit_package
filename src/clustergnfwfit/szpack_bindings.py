# to make shared library (libSZpack.so)
# https://stackoverflow.com/questions/655163/convert-a-static-library-to-a-shared-library

# insert at line 87
# g++ -shared -o ./libSZpack.so $? $(LIBS)

# problem with name mangling when calling functions:
# https://stackoverflow.com/questions/30983220/ctypes-error-attributeerror-symbol-not-found-os-x-10-7-5


import ctypes
from ctypes import c_int, c_double
import pathlib
import numpy as np

T0_CMB=2.726
Dn_DI_conversion=13.33914078*pow(T0_CMB, 3)

libSZpack_so_path = None
libSZpack_so_path = None

def verify_lib():
    """Verifies that SZpack shared library has been loaded.

    Raises:
        Exception: Raised when not loaded.
    """
    if libSZpack is None:
        raise Exception("Function load_lib must first be called.")


def load_lib(so_path):
    """Loads the SZpack shared library.

    Args:
        so_path (str): Path to .so file
    """
    global libSZpack_so_path
    global libSZpack
    libSZpack_so_path = so_path
    libname = pathlib.Path(libSZpack_so_path).absolute()
    libSZpack = ctypes.CDLL(libname)

    # set function arg / res types    
    libSZpack._Z20compute_SZ_signal_5Ddddddddd.argtypes = [c_double] * 8
    libSZpack._Z20compute_SZ_signal_5Ddddddddd.restype = c_double

    libSZpack._Z20compute_SZ_signal_5DPdiddddddd.argtypes = [np.ctypeslib.ndpointer(c_double), c_int] + [c_double] * 7
    libSZpack._Z20compute_SZ_signal_5DPdiddddddd.restype = None

    libSZpack._Z20compute_SZ_signal_3Ddddddddd.argtypes = [c_double] * 8
    libSZpack._Z20compute_SZ_signal_3Ddddddddd.restype = c_double

    libSZpack._Z28compute_SZ_signal_asymptoticdddddddii.argtypes = [c_double] * 7 + [c_int] * 2
    libSZpack._Z28compute_SZ_signal_asymptoticdddddddii.restype = c_double

    libSZpack._Z28compute_SZ_signal_asymptoticPdiddddddii.argtypes = [np.ctypeslib.ndpointer(c_double), c_int] + [c_double] * 6 + [c_int] * 2
    libSZpack._Z28compute_SZ_signal_asymptoticPdiddddddii.restype = None

    libSZpack._Z28compute_SZ_signal_CNSN_basisdddddddii.argtypes = [c_double] * 7 + [c_int] * 2
    libSZpack._Z28compute_SZ_signal_CNSN_basisdddddddii.restype = c_double

    libSZpack._Z32compute_SZ_signal_CNSN_basis_optdddddddiii.argtypes = [c_double] * 7 + [c_int] * 3
    libSZpack._Z32compute_SZ_signal_CNSN_basis_optdddddddiii.restype = c_double

    libSZpack._Z23compute_SZ_signal_comboddddddd.argtypes = [c_double] * 7
    libSZpack._Z23compute_SZ_signal_comboddddddd.restype = c_double

    # Do later, or not
    # libSZpack._Z28Dcompute_SZ_signal_combo_CMBdiiiddddRSt6vectorIdSaIdEE.argtypes = [ctypes.c_double] + [ctypes.c_int] * 3 + [ctypes.c_double] * 4

    libSZpack._Z29compute_SZ_signal_combo_meansdddddddd.argtypes = [c_double] * 8
    libSZpack._Z29compute_SZ_signal_combo_meansdddddddd.restype = c_double

    libSZpack._Z25compute_null_of_SZ_signalddddddd.argtypes = [c_double] * 7
    libSZpack._Z25compute_null_of_SZ_signalddddddd.restype = c_double

    libSZpack._Z32compute_SZ_signal_combo_means_exddddPdS_dd.argtypes = [c_double] * 4 + [c_double * 3] * 2 + [c_double] * 2
    libSZpack._Z32compute_SZ_signal_combo_means_exddddPdS_dd.restype = c_double


# Maybe add option for destructive processing of xo array

def output_distortion(x, Dn):
    x3_Dn = pow(x, 3) * Dn
    return x, x3_Dn, Dn_DI_conversion*x3_Dn

def compute_SZ_signal_5D_scalar(xo, Dtau, Te, betac, muc, betao, muo, eps_Int=1.0e-4):
    verify_lib()
    return libSZpack._Z20compute_SZ_signal_5Ddddddddd(*map(c_double, (xo, Dtau, Te, betac, muc, betao, muo, eps_Int)))

def compute_SZ_signal_5D_vector(xo, Dtau, Te, betac, muc, betao, muo, eps_Int=1.0e-4):
    verify_lib()
    xo = np.copy(xo)
    libSZpack._Z20compute_SZ_signal_5DPdiddddddd(np.ascontiguousarray(xo, np.float64), xo.size, Dtau, Te, betac, muc, betao, muo, eps_Int)
    return xo

# def compute_SZ_signal_5D_vector(xo, Dtau, Te, betac, muc, betao, muo, eps_Int=1.0e-4):
#     verify_lib()
#     res = []
#     for x in xo:
#         res.append(compute_SZ_signal_5D_scalar(x, Dtau, Te, betac, muc, betao, muo, eps_Int))
#     return res

def output_SZ_distortion_5D_vector(xo, Dtau, Te, betac, muc, betao, muo, eps_Int=1.0e-4):
    verify_lib()
    res = []
    signal = compute_SZ_signal_5D_vector(xo, Dtau, Te, betac, muc, betao, muo, eps_Int=1.0e-4)
    for k in range(len(signal)):
        dum = signal[k]
        res.append(output_distortion(xo[k], dum))
    return res

def compute_SZ_signal_3D(xo, Dtau, Te, betac, muc, betao, muo, eps_Int=1.0e-4):
    verify_lib()
    return libSZpack._Z20compute_SZ_signal_3Ddddddddd(*map(c_double, (xo, Dtau, Te, betac, muc, betao, muo, eps_Int)))


def compute_SZ_signal_asymptotic_scalar(xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order):
    verify_lib()
    return libSZpack._Z28compute_SZ_signal_asymptoticdddddddii(*map(c_double, (xo, Dtau, Te, betac, muc, betao, muo)), *map(c_int, (Te_order, betac_order)))

def compute_SZ_signal_asymptotic_vector(xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order):
    verify_lib()
    xo = np.copy(xo)
    libSZpack._Z28compute_SZ_signal_asymptoticPdiddddddii(np.ascontiguousarray(xo, np.float64), xo.size, xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order)
    return xo

# def compute_SZ_signal_asymptotic_vector(xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order):
#     verify_lib()
#     res = []
#     for x in xo:
#         res.append(compute_SZ_signal_asymptotic_scalar(x, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order))
#     return res

def output_SZ_distortion_asymptotic_vector(xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order):
    verify_lib()
    res = []
    signal = compute_SZ_signal_asymptotic_vector(xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order)
    for k in range(len(signal)):
        dum = signal[k]
        res.append(output_distortion(xo[k], dum))
    return res

def compute_SZ_signal_CNSN_basis(xo, Dtau, Te, betac, muc, betao, muo, Te_order, betac_order):
    verify_lib()
    return libSZpack._Z28compute_SZ_signal_CNSN_basisdddddddii(*map(c_double, (xo, Dtau, Te, betac, muc, betao, muo)), *map(c_int, (Te_order, betac_order)))


def compute_SZ_signal_CNSN_basis_opt(xo, Dtau, Te, betac, muc, betao, muo, kmax, betac_order, accuracy_level):
    verify_lib()
    return libSZpack._Z32compute_SZ_signal_CNSN_basis_optdddddddiii(*map(c_double, (xo, Dtau, Te, betac, muc, betao, muo)), *map(c_int, (kmax, betac_order, accuracy_level)))


def compute_SZ_signal_combo(xo, Dtau, Te, betac, muc, betao, muo):
    verify_lib()
    return libSZpack._Z23compute_SZ_signal_comboddddddd(*map(c_double, (xo, Dtau, Te, betac, muc, betao, muo)))

def compute_SZ_signal_combo_means(xo, 
                                # mean parameters
                                tau, TeSZ, betac_para,
                                # variances
                                omega, sigma, 
                                kappa, betac2_perp):
    verify_lib()
    return libSZpack._Z29compute_SZ_signal_combo_meansdddddddd(*map(c_double, (xo, tau, TeSZ, betac_para, omega, sigma, kappa, betac2_perp)))

def compute_null_of_SZ_signal(tau, TeSZ, betac_para, omega, sigma, kappa, betac2_perp):
    verify_lib()
    return libSZpack._Z25compute_null_of_SZ_signalddddddd(*map(c_double, (tau, TeSZ, betac_para, omega, sigma, kappa, betac2_perp)))

def compute_SZ_signal_combo_means_ex(xo, 
                                    # mean parameters
                                    tau, TeSZ, betac_para,
                                    # variances
                                    omega, sigma, 
                                    kappa, betac2_perp):
    verify_lib()
    return libSZpack._Z32compute_SZ_signal_combo_means_exddddPdS_dd(*map(c_double, (xo, tau, TeSZ, betac_para)),
                                                                    (c_double*3)(*omega), (c_double*3)(*sigma),
                                                                    *map(c_double, (kappa, betac2_perp)))


if __name__ == "__main__":
    import time

    load_lib('src/clustergnfwfit/SZpack.v1.1.1/libSZpack.so')

    start = time.time()
    x = 0.1
    x3 = pow(x, 3)
    ans = compute_SZ_signal_5D_scalar(0.1, 0.01, 15.33, 0.01, 1, 0.001241, 0, 0.0001)
    print(x, x3*ans)
    print(time.time() - start)

    start = time.time()
    xo = np.full((10), 0.1)
    ans = compute_SZ_signal_5D_vector(xo, 0.01, 15.33, 0.01, 1, 0.001241, 0, 0.0001)
    print(x, x3*ans)
    print(time.time() - start)
