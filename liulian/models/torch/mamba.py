"""Re-export from mamba_model to allow ``model: mamba`` in configs.

The actual implementation lives in :mod:`liulian.models.torch.mamba_model`
(named ``mamba_model`` to avoid import conflicts with the ``mamba-ssm``
package).  This shim enables the pipeline's dynamic import
``importlib.import_module('liulian.models.torch.mamba')`` to find the
``Model`` class.
"""

from liulian.models.torch.mamba_model import Model  # noqa: F401

__all__ = ['Model']
