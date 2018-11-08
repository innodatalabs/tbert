import torch
import numpy as np


def as_torch_dtype(numpy_dtype):
    '''Takes numpy dtype and returns corresponding torch dtype'''

    if numpy_dtype == np.int32:
        return torch.int32
    if numpy_dtype == np.int64:
        return torch.long
    if numpy_dtype == np.float32:
        return torch.float
    if numpy_dtype == np.float64:
        return torch.double
    assert False, (numpy_dtype, np.int32)


def compatible_dtypes(x, y):
    '''Returns True if tensor data types are compatible'''
    if x == y:
        return True
    if (x, y) == (torch.int32, torch.long) or (x, y) == (torch.long, torch.int32):
        return True
    if (x, y) == (torch.float, torch.double) or (x, y) == (torch.double, torch.long):
        return True


def assert_same(array, tensor, tolerance=1.e-5):
    '''Asserts that numpy array and torch tensor have essentially the same data'''
    if as_torch_dtype(array.dtype) != tensor.dtype:
        if not compatible_dtypes(as_torch_dtype(array.dtype), tensor.dtype):
            raise AssertionError('dtype mismatch: %r vs %r' % (array.dtype, tensor.dtype))

    if tuple(array.shape) != tuple(tensor.shape):
        raise AssertionError('shape mismatch: %r vs %r' % (array.shape, tensor.shape))

    array = torch.from_numpy(array).to(tensor.dtype)
    max_diff = torch.max(torch.abs(tensor - array))
    if max_diff > tolerance:
        print(array)
        print(tensor)
        print(torch.abs(tensor - array))
        raise AssertionError('values mismatch by %r' % max_diff.item())


def set_param_value(module, param_name, value):
    '''Sets the value of module parameter, checking shape and dtype'''
    x = module
    for part in param_name.split('.'):
        x = getattr(x, part)

    if x.data.dtype != value.dtype:
        raise ValueError('Can not assign value with dtype %r to %s having dtype %r at'
            % (value.dtype, param_name, x.data.dtype)
        )
    if x.data.shape != value.shape:
        raise ValueError('Can not assign value with shape %r to %s having shape %r'
            % (value.shape, param_name, x.data.shape)
        )

    x.data = value


def assign_params(module, vars, mapping, **fmt):
    '''Loads parameters from dictionary'''
    for param_name, item in mapping.items():
        var = vars[item['path'].format(**fmt)]
        if item.get('transpose'):
            var = var.T
        t = torch.FloatTensor(var)

        set_param_value(module, param_name, t)


