from __future__ import print_function
import mxnet as mx
from mxnet import nd as T
import numpy as np

from .common import _FLOATX, floatx, _EPSILON, image_dim_ordering, reset_uids, get_uid
from numbers import Number

_LEARNING_PHASE = 1
_EXECUTOR = None
_MODEL = None
_bind_values = {}


def set_model(model):
    global _MODEL
    _MODEL = model


def clear_session():
    reset_uids()
    _bind_values.clear()
    _EXECUTOR = None
    _MODEL = None


def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value


# VARIABLE MANIPULATION
def _typename(t):
    if t == np.float16:
        return 'float16'
    elif t == np.float32:
        return 'float32'
    elif t == np.float64:
        return 'float64'
    elif t == np.uint8:
        return 'uint8'
    elif t == np.uint16:
        return 'uint16'
    elif t == np.int16:
        return 'int16'
    elif t == np.int32:
        return 'int32'
    elif t == np.int64:
        return 'int64'
    else:
        raise TypeError('unknown type')


def is_sparse(tensor):
    return False


def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return np.float16
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    elif dtype == 'int16':
        return np.int16
    elif dtype == 'int32':
        return np.int32
    elif dtype == 'int64':
        return np.int64
    elif dtype == 'uint8':
        return np.int8
    elif dtype == 'uint16':
        return np.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)


def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor
    and returns it.

    # Arguments
        tensor: A tensor instance (potentially sparse).

    # Returns
        A dense tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
        >>> c = K.to_dense(b)
        >>> print(K.is_sparse(c))
        False
    ```
    """
    raise NotImplementedError


class KerasSymbol(object):
    def __init__(self, symbol):
        if not isinstance(symbol, mx.symbol.Symbol):
            raise TypeError
        self.symbol = symbol
        self._uses_learning_phase = False

    def bind(self, data):
        self.tensor = data
        if self.name in _bind_values:
            assert _bind_values[self.name].shape == data.shape, \
                "Redefinition of variable %s"%self.name
            assert _bind_values[self.name].dtype == data.dtype, \
                "Redefinition of variable %s"%self.name
            _bind_values[self.name][:] = data
        else:
            _bind_values[self.name] = data

    @property
    def name(self):
        return self.symbol.name

    @property
    def dtype(self):
        return self.get_type()

    @property
    def shape(self):
        return self.get_shape()

    def get_shape(self):
        if hasattr(self, 'tensor'):
            return self.tensor.shape
        else:
            _, out_shape, _ = self.symbol.infer_shape_partial()
            return out_shape[0]

    def get_type(self):
        if hasattr(self, 'tensor'):
            return _typename(self.tensor.dtype)
        else:
            _, out_type, _ = self.symbol.infer_type()
            t = out_type[0]
            return _typename(t)

    def __abs__(self):
        return KerasSymbol(mx.sym.abs(self.symbol))

    def __add__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_add(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(
                self.symbol + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_minus(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(self.symbol - other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __neg__(self):
        return KerasSymbol(self.symbol * (-1.0))

    def __div__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol / other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_div(
                    lhs=self.symbol,
                    rhs=other.symbol))

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol / other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_div(
                    lhs=self.symbol,
                    rhs=other.symbol))

    def __mul__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_mul(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(self.symbol * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # def __eq__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(self.symbol == other)
    #     else:
    #         return KerasSymbolCompare(
    #             mx.sym.broadcast_equal(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol), self, other)

    # def __gt__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(self.symbol > other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_greater(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __ge__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(self.symbol >= other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_greater_equal(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __lt__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(
    #             self.symbol < other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_lesser(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __le__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(
    #             self.symbol <= other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_lesser_equal(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __gt__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(
    #             self.symbol > other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_greater(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __ge__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(
    #             self.symbol >= other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_greater_equal(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __lt__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(
    #             self.symbol < other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_lesser(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    # def __le__(self, other):
    #     if isinstance(other, Number):
    #         return KerasSymbol(
    #             self.symbol <= other)
    #     else:
    #         return KerasSymbol(
    #             mx.sym.broadcast_lesser_equal(
    #                 lhs=self.symbol,
    #                 rhs=other.symbol))

    def __pow__(self, power, modulo=None):
        return KerasSymbol(self.symbol.__pow__(power))

    def __repr__(self):
        return self.symbol.name + ':[tensor=' + str(hasattr(self, 'tensor')) +\
            ' dtype=' + self.dtype + ']'

    def __str__(self):
        return "Symbol:" + self.symbol.name

    def __hash__(self):
        return hash(self.name)


class KerasSymbolCompare(KerasSymbol):
    def __init__(self, symbol, left, right):
        super(KerasSymbolCompare, self).__init__(symbol)
        self._left = left
        self._right = right

    def __bool__(self):
        return self._left.name == self._right.name


def KerasVariable(name, shape, dtype):
    if dtype is None:
        dtype = floatx()
    v = mx.sym.Variable(name, shape=shape, dtype=dtype)
    ret = KerasSymbol(v)
    return ret


def _autogen_name(prefix):
    return prefix + str(get_uid(prefix))


def variable(value, dtype=None, name=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.

    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> kvar.eval()
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if hasattr(value, 'tocoo'):
        raise NotImplementedError
    if name is None:
        name = _autogen_name('variable')
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    if isinstance(value, Number):
        value = np.array([value])
    ndarray = mx.nd.array(value, dtype=dtype)
    ret = KerasVariable(name, ndarray.shape, ndarray.dtype)
    ret.bind(ndarray)
    if isinstance(value, np.ndarray):
        ret._keras_shape = tuple([d if d != 0 else None for d in value.shape])
    elif hasattr(value, 'get_shape'):
        ret._keras_shape = tuple([d if d != 0 else None for d in map(int, value.get_shape())])
    ret._uses_learning_phase = False
    return ret


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph._keras_shape
        (2, 4, 5)
        >>> input_ph
        <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    if name is None:
        name = _autogen_name('placeholder')
    if not shape:
        if ndim:
            shape = tuple([0 for _ in range(ndim)])
    else:
        shape = tuple([0 if x is None else x for x in shape])
    sym = KerasVariable(name, shape=shape, dtype=dtype)
    sym._keras_shape = tuple([d if d != 0 else None for d in shape])
    sym._uses_learning_phase = False
    return sym


def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```
        # TensorFlow example
        >>> from keras import backend as K
        >>> tf_session = K.get_session()
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> input = keras.backend.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
        >>> K.shape(input)
        <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
        # To get integer shape (Instead, you can use K.int_shape(x))
        >>> K.shape(kvar).eval(session=tf_session)
        array([2, 2], dtype=int32)
        >>> K.shape(input).eval(session=tf_session)
        array([2, 4, 5], dtype=int32)
    ```
    """
    #   if hasattr(x, '_keras_shape'):
    #       return tuple([0 if x is None else x for x in x._keras_shape])
    if isinstance(x, KerasSymbol):
        return x.get_shape()
    else:
        return None


def int_shape(x):
    """Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(input)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    s = shape(x)
    if s is None:
        return None
    else:
        return tuple([i.__int__() for i in s])


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(input)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    s = shape(x)
    if s is None:
        return 0
    return len(s)


def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32_ref'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32_ref'
    ```
    """
    return x.dtype


def eval(x):
    """Evaluates the value of a variable.
    Returns a Numpy array.

    # Arguments
        x: A variable.

    # Returns
        A Numpy array.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
    ```
    """
    if isinstance(x, KerasSymbol):
        if hasattr(x, 'tensor'):
            if x.name in _bind_values and _MODEL is not None:
                _MODEL._sync_weights()
            ret = x.tensor.asnumpy()
        else:
            executor = x.symbol.simple_bind(mx.cpu(), grad_req='null')
            for v in executor.arg_dict:
                _bind_values[v].copyto(executor.arg_dict[v])
            outputs = executor.forward(is_train=_LEARNING_PHASE)
            ret = outputs[0].asnumpy()

        if ret.shape == (1,):
            return ret[0]
        return ret
    else:
        return x


def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.zeros((3,4))
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    value = mx.nd.zeros(shape, dtype=dtype)
    if name is None:
        name = _autogen_name('zeroinit')
    ret = KerasVariable(name, value.shape, value.dtype)
    ret.bind(value)
    return ret


def ones(shape, dtype=None, name=None):
    """Instantiates an all-ones tensor variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, filled with `1.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.ones((3,4))
        >>> K.eval(kvar)
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    value = mx.nd.ones(shape, dtype=dtype)
    if name is None:
        name = _autogen_name('oneinit')
    ret = KerasVariable(name, value.shape, value.dtype)
    ret.bind(value)
    return ret


def eye(size, dtype=None, name=None):
    """Instantiate an identity matrix and returns it.

    # Arguments
        size: Integer, number of rows/columns.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, an identity matrix.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.eye(3)
        >>> K.eval(kvar)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
    ```
    """
    value = mx.nd.array(np.eye(size, dtype=dtype))
    if name is None:
        name = _autogen_name('eyeinit')
    ret = KerasVariable(name, value.shape, value.dtype)
    ret.bind(value)
    return ret


def zeros_like(x, name=None):
    """Instantiates an all-zeros Keras variable
    of the same shape as another Keras variable or tensor and returns it.

    # Arguments
        x: Keras variable or Keras tensor.

    # Returns
        A Keras variable, filled with `0.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_zeros = K.zeros_like(kvar)
        >>> K.eval(kvar_zeros)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    value = mx.nd.zeros(shape(x), dtype=dtype(x))
    if name is None:
        name = _autogen_name('zerolikeinit')
    ret = KerasVariable(name, shape(x), dtype(x))
    ret.bind(value)
    return ret


def ones_like(x, name=None):
    """Instantiates an all-ones Keras variable
    of the same shape as another Keras variable or tensor and returns it.

    # Arguments
        x: Keras variable or tensor.

    # Returns
        A Keras variable, filled with `1.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_ones = K.ones_like(kvar)
        >>> K.eval(kvar_ones)
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
    ```
    """
    value = mx.nd.ones(shape(x), dtype(x))
    if name is None:
        name = _autogen_name('zerolikeinit')
    ret = KerasVariable(name, shape(x), dtype(x))
    ret.bind(value)
    return ret


def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
    """Instantiates an Keras variable filled with
    samples drawn from a uniform distribution and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output inteval.
        high: Float, upper boundary of the output interval.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        # TensorFlow example
        >>> kvar = K.random_uniform_variable((2,3), 0, 1)
        >>> kvar
        <tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
        >>> K.eval(kvar)
        array([[ 0.10940075,  0.10047495,  0.476143  ],
               [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    value = mx.random.uniform(low=low, high=high, dtype='float32', shape=shape)
    if dtype != np.float32:
        value = mx.nd.Cast(value, dtype=dtype)
    name = _autogen_name("uniform")
    ret = KerasVariable(name, value.shape, value.dtype)
    ret.bind(value)
    return ret


def random_normal_variable(shape, mean, scale, dtype=None,
                           name=None, seed=None):
    """Instantiates an Keras variable filled with
    samples drawn from a normal distribution and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        mean: Float, mean of the normal distribution.
        scale: Float, standard deviation of the normal distribution.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        # TensorFlow example
        >>> kvar = K.random_normal_variable((2,3), 0, 1)
        >>> kvar
        <tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
        >>> K.eval(kvar)
        array([[ 1.19591331,  0.68685907, -0.63814116],
               [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    value = mx.random.normal(loc=mean, scale=scale, dtype='float32', shape=shape)
    if dtype != np.float32:
        value = mx.nd.Cast(value, dtype=dtype)
    name = _autogen_name("normal")
    ret = KerasVariable(name, value.shape, value.dtype)
    ret.bind(value)
    return ret


def count_params(x):
    """Returns the number of scalars in a Keras variable.

    # Arguments
        x: Keras variable.

    # Returns
        Integer, the number of scalars in `x`.

    # Example
    ```python
        >>> kvar = K.zeros((2,3))
        >>> K.count_params(kvar)
        6
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    shape = x.get_shape()
    return np.prod([shape[i] for i in range(len(shape))])


def cast(x, dtype):
    """Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder((2, 3), dtype='float32')
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>    ```
    """
    if isinstance(x, KerasSymbol):
        return KerasSymbol(
            mx.sym.Cast(data=x.symbol, dtype=dtype))
    else:
        return x.astype(dtype)


# UPDATES OPS

# Don't need
def update(x, new_x):

    raise NotImplementedError


# Don't need
def update_add(x, increment):
    raise NotImplementedError


# Don't need
def update_sub(x, decrement):
    raise NotImplementedError


# Don't need
def moving_average_update(variable, value, momentum):
    return variable, variable * momentum + value * (1. - momentum)


# LINEAR ALGEBRA

def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a ND tensor
    with a ND tensor, it reproduces the Theano behavior.
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

    # Arguments
        x: Tensor or variable.
    dtype = _convert_string_dtype(dtype)
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.

    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
    ```

    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    """
    return KerasSymbol(mx.sym.dot(lhs=x.symbol, rhs=y.symbol))


def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x, y: Keras tensors or variables with `ndim >= 2`
            (With TensorFlow backend, `batch_dot()` only supports `ndim >= 3`)
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
        of `x.dot(y.T)`, although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
        If `axes` is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in `x`'s shape and `y`'s shape:

        * `x.shape[0]` : 100 : append to output shape
        * `x.shape[1]` : 20 : do not append to output shape,
            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
        * `y.shape[0]` : 100 : do not append to output shape,
            always ignore first dimension of `y`
        * `y.shape[1]` : 30 : append to output shape
        * `y.shape[2]` : 20 : do not append to output shape,
            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
        `output_shape` = `(100, 30)`

    ```python
        >>> x_batch = K.ones(shape=(32, 20, 1))
        >>> y_batch = K.ones(shape=(32, 30, 20))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```
    """
    if ndim(x) == ndim(y) == 2:
        assert axes is None or axes == 1 or tuple(axes) == (1, 1)
        axes = (2, 1)
        x = expand_dims(x, dim=1)
        y = expand_dims(y, dim=2)
        extra = True
    else:
        assert ndim(x) == ndim(y) == 3, "Only support 2d or 3d tensors for now"
        extra = False

    if isinstance(axes, Number):
        axes = (axes, axes)
    if axes is None:
        Ta = Tb = False
    else:
        Ta = not bool(axes[0] - 1)
        Tb = bool(axes[1] - 1)

    ret = KerasSymbol(mx.sym.batch_dot(lhs=x.symbol, rhs=y.symbol,
                                       transpose_a=Ta, transpose_b=Tb))
    if extra:
        ret = squeeze(ret, 2)

    return ret

def transpose(x):
    """Transposes a tensor and returns it.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.

    # Examples
    ```python
        >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
        >>> K.eval(var)
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> var_transposed = K.transpose(var)
        >>> K.eval(var_transposed)
        array([[ 1.,  4.],
               [ 2.,  5.],
               [ 3.,  6.]], dtype=float32)
    ```

    ```python
        >>> input = K.placeholder((2, 3))
        >>> input
        <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
        >>> input_transposed = K.transpose(input)
        >>> input_transposed
        <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

    ```
    """
    return KerasSymbol(
        mx.sym.transpose(data=x.symbol))


def gather(reference, indices):
    """Retrieves the elements of indices `indices`
    in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    raise NotImplementedError


# ELEMENT-WISE OPERATIONS

def _normalize_axis(axis, ndim):
    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list):
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    elif axis is None:
        return ()
    else:
        if axis < 0:
            axis = axis % ndim
    return axis


def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find maximum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.max(data=x.symbol, axis=axis, keepdims=keepdims))


def min(x, axis=None, keepdims=False):
    """Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find minimum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.min(data=x.symbol, axis=axis, keepdims=keepdims))


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to sum over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.sum(data=x.symbol, axis=axis, keepdims=keepdims))


def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.prod(data=x.symbol, axis=axis, keepdims=keepdims))


def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol

    v = _var(x, axis, keepdims)
    return KerasSymbol(v)


def _var(x, axis=None, keepdims=False):
    mean_input = mx.sym.mean(data=x, axis=axis, keepdims=True)
    centered_input = mx.sym.broadcast_minus(lhs=x, rhs=mean_input)
    v = mx.sym.mean(data=(centered_input ** 2), axis=axis, keepdims=keepdims)
    return v


def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    v = var(x, axis=axis, keepdims=keepdims)
    ret = mx.sym.sqrt(data=v.symbol)
    return KerasSymbol(ret)


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keep_dims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    if axis == []:
        return x
    axis = _normalize_axis(axis, ndim(x))

    if axis is not None:
        ret = mx.sym.mean(data=x.symbol, axis=axis, keepdims=keepdims)
    else:
        ret = mx.sym.mean(data=x.symbol, keepdims=keepdims)
    return KerasSymbol(ret)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    sum0 = mx.sym.sum_axis(x, axis=axis, keepdims=keepdims)
    # tmp = KerasSymbol(sum0)
    # bg = mx.sym.broadcast_greater(lhs=sum0, rhs=0)
    # bg_tmp = KerasSymbol()
    return KerasSymbol(sum0 > 0)


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    abs = mx.sym.abs(data=x)
    min = mx.sym.min_axis(data=abs, axis=axis, keepdims=keepdims)
    return KerasSymbol(mx.sym.broadcast_greater(min, 0))


def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A tensor.
    """
    axis = _normalize_axis(axis, ndim(x))
    ret = mx.sym.argmax(data=x.symbol, axis=axis)
    return KerasSymbol(ret)


def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A tensor.
    """
    axis = _normalize_axis(axis, ndim(x))
    ret = mx.sym.argmin(data=x.symbol, axis=axis)
    return KerasSymbol(ret)


def square(x):
    """Element-wise square.
    """
    return KerasSymbol(mx.sym.square(data=x.symbol))


def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.abs(data=x.symbol))


def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    ret = mx.sym.Activation(data=x.symbol, act_type='relu')
    ret = mx.sym.sqrt(data=ret)
    return KerasSymbol(ret)


def exp(x):
    """Element-wise exponential.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.exp(data=x.symbol))


def log(x):
    """Element-wise log.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.log(data=x.symbol))


def round(x):
    """Element-wise rounding to the closest integer.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.round(data=x.symbol))


def sign(x):
    """Element-wise sign.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.sign(data=x.symbol))


def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(a, KerasSymbol):
        a = a.symbol
    return KerasSymbol(mx.sym.pow(base=x, exp=a))


def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Returns
        A tensor.
    """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    min_value = np.float32(min_value)
    max_value = np.nan_to_num(np.float32(max_value))

    return KerasSymbol(mx.sym.clip(data=x.symbol, a_min=min_value, a_max=max_value))


def equal(x, y):
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.broadcast_equal(lhs=x, rhs=y))


def not_equal(x, y):
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(
        x.symbol.__ne__(y))


def greater(x, y):
    """Element-wise truth value of (x > y).

    # Returns
        A bool tensor.
    """
    return KerasSymbol(x.symbol > y.symbol)


def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Returns
        A bool tensor.
    """
    return KerasSymbol(x.symbol >= y.symbol)


def lesser(x, y):
    """Element-wise truth value of (x < y).

    # Returns
        A bool tensor.
    """
    return KerasSymbol(x.symbol < y.symbol)


def lesser_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Returns
        A bool tensor.
    """
    return KerasSymbol(x.symbol <= y.symbol)


def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.maximum(left=x, right=y))


def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.minimum(left=x, right=y))


def sin(x):
    """Computes sin of x element-wise.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.sin(data=x.symbol))


def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.cos(data=x.symbol))


def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    original_x = x
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol
    # normal, mean, var = mx.sym.BatchNorm(data=x, gamma=gamma.symbol, beta=beta.symbol, output_mean_var=True)

    mean = mx.sym.mean(data=x, axis=reduction_axes, keepdims=False)
    var = _var(x, axis=reduction_axes, keepdims=False)

    list_axe = list(range(ndim(original_x))[:-1])
    if sorted(reduction_axes) == list_axe:
        normed = batch_normalization(x, mean, var,
                                     beta, gamma,
                                     epsilon)
    else:
        # need broadcasting
        target_shape = []
        for axis in range(ndim(original_x)):
            if axis in reduction_axes:
                target_shape.append(1)
            else:
                target_shape.append(original_x.shape[axis])
        target_shape = tuple(target_shape)

        broadcast_mean = mx.sym.Reshape(data=mean, shape=target_shape)
        broadcast_var = mx.sym.Reshape(data=var, shape=target_shape)
        broadcast_gamma = mx.sym.Reshape(data=gamma, shape=target_shape)
        broadcast_beta = mx.sym.Reshape(data=beta, shape=target_shape)
        normed = batch_normalization(x, broadcast_mean, broadcast_var,
                                     broadcast_beta, broadcast_gamma,
                                     epsilon)

    return normed, KerasSymbol(mean), KerasSymbol(var)


def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Apply batch normalization on x given mean, var, beta and gamma.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(mean, KerasSymbol):
        mean = mean.symbol
    if isinstance(var, KerasSymbol):
        var = var.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    std = mx.sym.sqrt(data=var + epsilon)

    minus0 = mx.sym.broadcast_minus(x, mean)
    div0 = mx.sym.broadcast_div(gamma, std)
    mul0 = mx.sym.broadcast_mul(minus0, div0)
    rval = mx.sym.broadcast_plus(mul0, beta)
    return KerasSymbol(rval)


# SHAPE OPERATIONS
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(tensors[0])
    tensors = [t.symbol for t in tensors]
    return KerasSymbol(mx.sym.Concat(*tensors, dim=axis))


def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Reshape(data=x.symbol, shape=shape))


def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        pattern: should be a tuple of
            dimension indices, e.g. (0, 2, 1).

    # Returns
        A tensor.
    """
    raise KerasSymbol(mx.sym.transpose(x.symbol, axes=pattern))


def resize_images(X, height_factor, width_factor, dim_ordering):
    """Resizes the images contained in a 4D tensor of shape
    - `[batch, channels, height, width]` (for 'th' dim_ordering)
    - `[batch, height, width, channels]` (for 'tf' dim_ordering)
    by a factor of `(height_factor, width_factor)`. Both factors should be
    positive integers.

    # Returns
        A tensor.
    """
    raise NotImplementedError


def resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering):
    """Resizes the volume contained in a 5D tensor of shape
    - `[batch, channels, depth, height, width]` (for 'th' dim_ordering)
    - `[batch, depth, height, width, channels]` (for 'tf' dim_ordering)
    by a factor of `(depth_factor, height_factor, width_factor)`.
    All three factors should be positive integers.

    # Returns
        A tensor.
    """
    raise NotImplementedError


def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Returns
        A tensor.
    """
    raise NotImplementedError


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Returns
        A tensor.
    """
    raise NotImplementedError


def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1-D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.
    """
    dtype = np.dtype(dtype)
    return KerasSymbol(mx.sym.arange(start=start, stop=stop, step=step, dtype=dtype))


def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    """
    raise NotImplementedError


def flatten(x):
    """Flatten a tensor.

    # Returns
        A tensor, reshaped into 1-D
    """
    raise NotImplementedError


def batch_flatten(x):
    """Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.

    In other words, it flattens each data samples of a batch.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Flatten(data=x.symbol))


def expand_dims(x, dim=-1):
    """Adds a 1-sized dimension at index "dim".

    # Returns
        A tensor with expended dimensions.
    """
    if dim < 0:
        dim %= len(x.get_shape())
    if isinstance(x, KerasSymbol):
        shape = list(x.get_shape())
        x = x.symbol
        return KerasSymbol(mx.sym.expand_dims(x, axis=dim))


def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    shape = list(x.get_shape())
    assert shape.pop(axis) == 1, "Can only squeeze size 1 dimension"

    if isinstance(x, KerasSymbol):
        x = x.symbol
        return KerasSymbol(mx.sym.Reshape(data=x, shape=tuple(shape)))


def temporal_padding(x, padding=1):
    """Pads the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    # Returns
        A padded 3D tensor.
    """
    return asymmetric_temporal_padding(x, padding, padding)


def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    """Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.

    # Returns
        A padded 3D tensor.
    """

    return KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant',
                                  constant_value=0,
                                  pad_width=(0, 0, left_pad, right_pad, 0, 0)))


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='default'):
    """Pads the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.

    # Returns
        A padded 4D tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'th':
        pattern = (0, 0, 0, 0,
                   padding[0], padding[0], padding[1], padding[1])
    else:
        pattern = (0, 0,
                   padding[0], padding[0], padding[1], padding[1],
                   0, 0)
    return KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant',
                                  constant_value=0,
                                  pad_width=pattern))


def asymmetric_spatial_2d_padding(x, top_pad=1, bottom_pad=1,
                                  left_pad=1, right_pad=1,
                                  dim_ordering='default'):
    """Pad the rows and columns of a 4D tensor
    with "top_pad", "bottom_pad", "left_pad", "right_pad" (resp.) zeros
    rows on top, bottom; cols on left, right.

    # Returns
        A padded 4D tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'th':
        pattern = (0, 0,
                   0, 0,
                   top_pad, bottom_pad,
                   left_pad, right_pad)
    else:
        pattern = (0, 0,
                   top_pad, bottom_pad,
                   left_pad, right_pad,
                   0, 0)
    return KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant',
                                  constant_value=0,
                                  pad_width=pattern))


def spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='default'):
    """Pads 5D tensor with zeros for the depth, height, width dimension with
    "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right

    For 'tf' dim_ordering, the 2nd, 3rd and 4th dimension will be padded.
    For 'th' dim_ordering, the 3rd, 4th and 5th dimension will be padded.

    # Returns
        A padded 5D tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'th':
        pattern = (
            0, 0,
            0, 0,
            padding[0], padding[0],
            padding[1], padding[1],
            padding[2], padding[2]
        )
    else:
        pattern = (
            0, 0,
            padding[0], padding[0],
            padding[1], padding[1],
            padding[2], padding[2],
            0, 0
        )
    return KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant',
                                  constant_value=0,
                                  pad_width=pattern))


def stack(x):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    raise NotImplementedError


def one_hot(indices, nb_classes):
    """Input: nD integer tensor of shape `(batch_size, dim1, dim2, ... dim(n-1))`
    Output: (n + 1)D one hot representation of the input
    with shape `(batch_size, dim1, dim2, ... dim(n-1), nb_classes)`

    # Returns
        The one-hot tensor.
    """
    raise NotImplementedError


def reverse(x, axes):
    """Reverse a tensor along the the specified axes

    # Returns
        A tensor.
    """
    raise NotImplementedError


# VALUE MANIPULATION
def get_value(x):
    """Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    """
    return eval(x)


def batch_get_value(xs):
    """Returns the value of more than one tensor variable.

    # Arguments
        x: list of variables.

    # Returns
        A list of Numpy arrays.
    """
    return [get_value(x) for x in xs]


def set_value(x, value):
    """Sets the value of a variable,
    from a Numpy array. It returns `None`.
    """
    if _MODEL is not None:
        _MODEL._sync_weights()
    if isinstance(value, Number):
        value = [value]
    x.bind(mx.nd.array(value))
    if _MODEL is not None:
        _MODEL._set_weights()


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.
    It returns `None`.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    if _MODEL is not None:
        _MODEL._sync_weights()

    for p, w in tuples:
        if isinstance(w, Number):
            w = [w]
        p.bind(mx.nd.array(w))

    if _MODEL is not None:
        _MODEL._set_weights()


def get_variable_shape(x):
    """Returns shape of a variable.

    # Arguments
        A variable.

    # Returns
        A tuple of integers.
    """
    return x.shape


def print_tensor(x, message=''):
    """Print the message and the tensor when evaluated and return the same
    tensor.
    """
    print(message, eval(x))


def group(variables):
    return KerasSymbol(mx.sym.Group([i.symbol for i in variables]))


def make_loss(variables):
    return KerasSymbol(mx.sym.MakeLoss(variables.symbol))


# GRAPH MANIPULATION
class Function(object):
    def __init__(self, inputs, output, updates=[], **kwargs):
        self.output = output
        self.updates = updates
        if isinstance(inputs[-1], Number):
            self.is_train = inputs[-1]
            self.inputs = inputs[:-1]
        else:
            self.inputs = inputs
            self.is_train = _LEARNING_PHASE

    def __call__(self, inputs):
        ret_outputs = []
        if isinstance(inputs[-1], Number):
            self.is_train = inputs[-1]
            inputs = inputs[:-1]
        for x in self.output:
            data = {k.name: v for k, v in zip(self.inputs, inputs)}
            data = dict(data, **_bind_values)
            args = x.symbol.list_arguments()
            data_shapes = {k.name: v.shape for k, v in zip(self.inputs, inputs) if k.name in args}
            executor = x.symbol.simple_bind(mx.cpu(), grad_req='null', **data_shapes)
            for v in executor.arg_dict:
                if v in data:
                    executor.arg_dict[v][:] = data[v]
            outputs = executor.forward(is_train=self.is_train)
            ret_outputs.append(outputs[0].asnumpy())
        return ret_outputs


def function(inputs, outputs, updates=[], **kwargs):
    return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
    """Returns the gradients of `variables` (list of tensor variables)
    with regard to `loss`.
    """
    raise NotImplementedError


def stop_gradient(variables):
    """Returns `variables` but with zero gradient with respect to every other
    variables.
    """
    return KerasSymbol(mx.sym.BlockGrad(variables.symbol))


# CONTROL FLOW
def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: with TensorFlow the RNN is always unrolled, but with Theano you
            can use this boolean flag to unroll the RNN.
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                entry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.
    """
    raise NotImplementedError


def switch(condition, then_expression, else_expression):
    """Switches between two operations
    depending on a scalar value (`int` or `bool`).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.
    """
    raise NotImplementedError


def in_train_phase(x, alt):
    """Selects `x` in train phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    """
    if learning_phase() is 1:
        if isinstance(x, KerasSymbol):
            return x
        return x()
    if learning_phase() is 0:
        if isinstance(x, KerasSymbol):
            return alt
        return alt()
    raise AssertionError("Learning phase must be 0 or 1")


def in_test_phase(x, alt):
    '''Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    '''
    if learning_phase() is 1:
        return alt()
    elif learning_phase() is 0:
        return x()
    raise AssertionError("Learning phase must be 0 or 1")


def relu(x, alpha=0., max_value=None):
    """Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    """
    if isinstance(alpha, KerasSymbol):
        alpha = eval(alpha)
    if not isinstance(alpha, Number):
        raise NotImplementedError
    if alpha != 0.:
        if isinstance(x, KerasSymbol):
            x = x.symbol
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        ret = f1 * x + f2 * mx.sym.abs(x)
    else:
        ret = mx.sym.Activation(data=x.symbol,
                                act_type='relu')
    if max_value is not None:
        ret = mx.sym.minimum(ret, max_value)
    return KerasSymbol(ret)


def elu(x, alpha=1.):
    """Exponential linear unit.

    # Arguments
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.LeakyReLU(data=x.symbol, act_type='elu', slope=alpha))


def softmax(x):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(
        mx.sym.SoftmaxActivation(data=x.symbol))


def softplus(x):
    """Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Activation(data=x.symbol, act_type='softrelu'))


def softsign(x):
    """Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(
        x.symbol / (1 + mx.sym.abs(x.symbol)))


def categorical_crossentropy(output, target, from_logits=False):
    assert not from_logits
    axis = ndim(output) - 1
    output = output.symbol
    output = mx.sym.clip(output, a_min=_EPSILON, a_max=1. - _EPSILON)
    output = - mx.sym.sum(target.symbol * mx.sym.log(output), axis=axis)
    return KerasSymbol(output)


def sparse_categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy between an output tensor
    and a target tensor, where the target is an integer tensor.
    """
    output = mx.sym.softmax_cross_entropy(output.symbol, target.symbol)
    return KerasSymbol(output)


def binary_crossentropy(output, target, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        output: A tensor.
        target: A tensor with the same shape as `output`.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    assert not from_logits
    output = output.symbol
    output = mx.sym.clip(output, a_min=_EPSILON, a_max=1. - _EPSILON)
    output = -(target.symbol * mx.sym.log(output) + (1.0 - target.symbol) * mx.sym.log(1.0 - output))
    return KerasSymbol(output)


def sigmoid(x):
    """Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Activation(data=x.symbol, act_type='sigmoid'))


def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(
        mx.sym.clip(data=(0.2 * x.symbol + 0.5), a_min=0, a_max=1))


def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.tanh(data=x.symbol))


def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random,
    while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.
    # Returns
        A tensor.
    """
    return KerasSymbol(
        mx.sym.Dropout(data=x.symbol, p=level))


def l2_normalize(x, axis):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: input tensor.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(x)
    norm = mx.sym.sqrt(data=mx.sym.sum(data=mx.sym.square(data=x.symbol), axis=axis, keepdims=True))
    return KerasSymbol(mx.sym.broadcast_div(x.symbol, norm))


def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`

    # Arguments
        predictions: A tensor of shape `batch_size` x classes and type `float32`.
        targets: A tensor of shape batch_size and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.

    # Returns
        A tensor of shape `batch_size` and type `bool`. `output_i` is `True` if
        `targets_i` is within top-k values of `predictions_i`
    """
    return KerasSymbol(
        mx.sym.Cast(
            data=mx.sym.topk(data=predictions.symbol, k=k, ret_typ='mask'),
            dtype='uint8'))


# CONVOLUTIONS
def _preprocess_conv2d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = KerasSymbol(mx.sym.transpose(x.symbol, axes=(0, 3, 1, 2)))
    return x


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = KerasSymbol(mx.sym.transpose(kernel.symbol, axes=(3, 2, 0, 1)))
    return kernel


def _postprocess_conv2d_output(x, dim_ordering):
    if dim_ordering == 'tf':
        x = KerasSymbol(mx.sym.transpose(x.symbol, axes=(0, 2, 3, 1)))
    return x


def _calculation_pad_same(input_shape, kernel, strides, dilation):
    out_size = np.ceil(float(input_shape) / float(strides))
    pad_along = ((out_size - 1) * strides + kernel - input_shape)
    return int(pad_along / 2)


def _calculation_pad_full(input_shape, kernel_shape, strides, dilation):
    dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    pad = dil_kernel_shape - 1
    return pad


def _preprocess_border_mode(border_mode, input_shape, kernel, strides, dilation):
    if border_mode == 'same':
        padding = (_calculation_pad_same(input_shape[2], kernel[0], strides[0], dilation[0]),
                   _calculation_pad_same(input_shape[3], kernel[1], strides[1], dilation[1]))
    elif border_mode == 'full':
        padding = (_calculation_pad_full(input_shape[2], kernel[0], strides[0], dilation[0]),
                   _calculation_pad_full(input_shape[3], kernel[1], strides[1], dilation[1]))
    elif border_mode == 'valid':
        padding = (0, 0)
    else:
        raise ValueError('Invalid border mode:', border_mode)
    return padding


def conv1d(x, kernel, stride=1, border_mode='valid',
           image_shape=None, filter_shape=None):
    """1D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: stride integer.
        border_mode: string, `"same"` or `"valid"`.

    # Returns
        A tensor, result of 1D convolution.
    """
    raise NotImplementedError


def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering='default',
           image_shape=None, filter_shape=None, filter_dilation=(1, 1)):
    """2D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of 2D convolution.
    """

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    layout_kernel, nb_filter = _layout_kernel2("th", kernel.shape)
    padding = _preprocess_border_mode(border_mode, x.shape, layout_kernel, strides, filter_dilation)
    s = mx.sym.Convolution(data=x.symbol, name=kernel.name, kernel=layout_kernel, stride=strides, pad=padding,
                           num_filter=nb_filter, weight=kernel.symbol, no_bias=True)
    out = _postprocess_conv2d_output(KerasSymbol(s), dim_ordering)
    return out


def deconv2d(x, kernel, output_shape, strides=(1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    """2D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of transposed 2D convolution.
    """
    layout_kernel, nb_filter = _layout_kernel2(dim_ordering, kernel.shape)
    s = mx.sym.Deconvolution(data=x.symbol, name=kernel.name, kernel=layout_kernel, stride=strides,
                             num_filter=nb_filter, weight=kernel.symbol, no_bias=True, target_shape=output_shape)
    return KerasSymbol(s)


def atrous_conv2d(x, kernel, rate=1,
                  border_mode='valid',
                  dim_ordering='default',
                  image_shape=None, filter_shape=None):
    """Atrous 2D convolution. Also as known as dilated convolution.

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        rate: integer > 0, the sample stride.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of atrous transposed 2D convolution.
    """
    raise NotImplementedError


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     border_mode='valid', dim_ordering='default'):
    """2-D convolution with separable filters.
    """
    raise NotImplementedError


def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='default',
           volume_shape=None, filter_shape=None):
    """3D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of 3D convolution.
    """
    layout_kernel, nb_filter = _layout_kernel3(dim_ordering, kernel.shape)
    s = mx.sym.Convolution(data=x.symbol, name=kernel.name, kernel=layout_kernel, stride=strides,
                           num_filter=nb_filter, weight=kernel.symbol, no_bias=True)
    return KerasSymbol(s)


def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering='default',
           pool_mode='max'):
    """2D Pooling.

    # Arguments
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of `"valid"`, `"same"`.
        dim_ordering: one of `"th"`, `"tf"`.
        pool_mode: one of `"max"`, `"avg"`.

    # Returns
        A tensor, result of 2D pooling.
    """
    s = mx.sym.Pooling(data=x.symbol, kernel=pool_size, pool_type=pool_mode, pooling_convention=border_mode,
                       stride=strides)
    return KerasSymbol(s)


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='default', pool_mode='max'):
    """3D Pooling.

    # Arguments
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        border_mode: one of `"valid"`, `"same"`.
        dim_ordering: one of `"th"`, `"tf"`.
        pool_mode: one of `"max"`, `"avg"`.

    # Returns
        A tensor, result of 3D pooling.
    """
    s = mx.sym.Pooling(data=x.symbol, kernel=pool_size, pool_type=pool_mode, pooling_convention=border_mode,
                       stride=strides)
    return KerasSymbol(s)


def random_normal(shape, mean=0.0, std=1.0, dtype=None, seed=None):
    """Returns a tensor with normal distribution

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: A float, mean of the normal distribution to draw samples.
        std: A float, standard deviation of the normal distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    name = _autogen_name('normal')
    sym = mx.sym.normal(loc=mean, scale=scale, shape=shape, dtype='float32', name=name)
    if dtype != np.float32:
        sym = mx.sym.Cast(data=sym, dtype=dtype)
    ret = KerasSymbol(sym)
    return ret


def random_uniform(shape, low=0.0, high=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        low: A float, lower boundary of the uniform distribution
            to draw samples.
        high: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    name = _autogen_name('uniform')
    sym = mx.sym.uniform(low=low, high=high, shape=shape, dtype='float32', name=name)
    if dtype != np.float32:
        sym = mx.sym.Cast(data=sym, dtype=dtype)
    ret = KerasSymbol(sym)
    return ret


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with binomlai distribution

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomlai distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    raise NotImplementedError


# CTC
def ctc_label_dense_to_sparse(labels, label_lengths):
    raise NotImplementedError


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    raise NotImplementedError


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
               top_paths=1):
    """Decodes the output of a softmax using either
       greedy (also known as best path) or a constrained dictionary
       search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)` containing the prediction,
                or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
                each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`. This does
                not use a dictionary
        beam_width: if `greedy` is `false`: a beam search decoder will be used
                with a beam of this width
        top_paths: if `greedy` is `false`: how many of the most probable paths will be returned

    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that contains
                the decoded sequence. If `false`, returns the `top_paths` most probable
                decoded sequences. Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains the log probability of each decoded sequence
    """
    raise NotImplementedError


# HIGH ORDER FUNCTIONS
def map_fn(fn, elems, name=None):
    """Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph

    # Returns
        Tensor with first dimension equal to the elems and second depending on
        fn
    """
    raise NotImplementedError


def foldl(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[0]` in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Same type and shape as initializer
    """
    raise NotImplementedError


def foldr(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from right to left.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[-1]` in case of None)
        name: A string name for the foldr node in the graph

    # Returns
        Same type and shape as initializer
    """
    raise NotImplementedError


def _layout_kernel2(dim_ordering, kernel):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering == 'th':
        layout_kernel = (kernel[2], kernel[3])
        nb_filter = kernel[0]
    elif dim_ordering == 'tf':
        layout_kernel = (kernel[0], kernel[1])
        nb_filter = kernel[3]
    else:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))
    return layout_kernel, nb_filter


def _layout_kernel3(dim_ordering, kernel):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering == 'th':
        layout_kernel = (kernel[2], kernel[3])
        nb_filter = kernel[0]
    elif dim_ordering == 'tf':
        layout_kernel = (kernel[0], kernel[1])
        nb_filter = kernel[3]
    else:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))
    return layout_kernel, nb_filter
