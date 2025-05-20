from cupyx.scipy.linalg import _decomp_interpolative as _backend
import cupy as cp
import warnings
from cupy import asarray, ascontiguousarray

__all__ = [
    'estimate_rank',
    'estimate_spectral_norm',
    'estimate_spectral_norm_diff',
    'id_to_svd',
    'interp_decomp',
    'rand',
    'reconstruct_interp_matrix',
    'reconstruct_matrix_from_id',
    'reconstruct_skel_matrix',
    'seed',
    'svd',
]

_DTYPE_ERROR = ValueError("invalid input dtype (input must be float64 or complex128)")
_TYPE_ERROR = TypeError("invalid input type (must be array or LinearOperator)")

def _C_contiguous_copy(A):
    """
    Same as np.ascontiguousarray, but ensure a copy
    """
    A = asarray(A)
    if A.flags.c_contiguous:
        A = A.copy()
    else:
        A = ascontiguousarray(A)
    return A

def _is_real(A):
    try:
        if A.dtype == cp.complex128:
            return False
        elif A.dtype == cp.float64:
            return True
        else:
            raise _DTYPE_ERROR
    except AttributeError as e:
        raise _TYPE_ERROR from e

def interp_decomp(A, eps_or_k, rand=True, rng=None):
    from scipy.sparse.linalg import LinearOperator
    rng = cp.random.default_rng(rng)
    real = _is_real(A)

    if isinstance(A, cp.ndarray):
        A = _C_contiguous_copy(A)
        if eps_or_k < 1:
            eps = eps_or_k
            if rand:
                if real:
                    k, idx, proj = _backend.iddp_aid(A, eps, rng=rng)
                else:
                    k, idx, proj = _backend.idzp_aid(A, eps, rng=rng)
            else:
                if real:
                    k, idx, proj = _backend.iddp_id(A, eps)
                else:
                    k, idx, proj = _backend.idzp_id(A, eps)
            return k, idx, proj
        else:
            k = int(eps_or_k)
            if rand:
                if real:
                    idx, proj = _backend.iddr_aid(A, k, rng=rng)
                else:
                    idx, proj = _backend.idzr_aid(A, k, rng=rng)
            else:
                if real:
                    idx, proj = _backend.iddr_id(A, k)
                else:
                    idx, proj = _backend.idzr_id(A, k)
            return idx, proj
    elif isinstance(A, LinearOperator):

        if eps_or_k < 1:
            eps = eps_or_k
            if real:
                k, idx, proj = _backend.iddp_rid(A, eps, rng=rng)
            else:
                k, idx, proj = _backend.idzp_rid(A, eps, rng=rng)
            return k, idx, proj
        else:
            k = int(eps_or_k)
            if real:
                idx, proj = _backend.iddr_rid(A, k, rng=rng)
            else:
                idx, proj = _backend.idzr_rid(A, k, rng=rng)
            return idx, proj
    else:
        raise _TYPE_ERROR