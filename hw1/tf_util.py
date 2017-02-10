import numpy as np
import tensorflow as tf # pylint: ignore-module
#import builtins
import functools
import copy
import os
import collections

# ================================================================
# Import all names into common namespace
# ================================================================

clip = tf.clip_by_value

# Make consistent with numpy
# ----------------------------------------

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)
def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))
def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def concatenate(arrs, axis=0):
    return tf.concat(axis, arrs)
def argmax(x, axis=None):
    return tf.argmax(x, dimension=axis)

def switch(condition, then_expression, else_expression):
    '''Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    '''
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

# Extras
# ----------------------------------------
def l2loss(params):
    if len(params) == 0:
        return tf.constant(0.0)
    else:
        return tf.add_n([sum(tf.square(p)) for p in params])
def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
def categorical_sample_logits(X):
    # https://github.com/tensorflow/tensorflow/issues/456
    U = tf.random_uniform(tf.shape(X))
    return argmax(X - tf.log(-tf.log(U)), axis=1)

# ================================================================
# Global session
# ================================================================

def get_session():
    return tf.get_default_session()

def single_threaded_session():
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    return tf.Session(config=tf_config)

def make_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


ALREADY_INITIALIZED = set()
def initialize():
    new_variables = set(tf.all_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.initialize_variables(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def eval(expr, feed_dict=None):
    if feed_dict is None: feed_dict = {}
    return get_session().run(expr, feed_dict=feed_dict)

def set_value(v, val):
    get_session().run(v.assign(val))

def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(get_session(), fname)

# ================================================================
# Model components
# ================================================================


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = intprod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer,
                            collections=collections)

        if summary_tag is not None:
            tf.image_summary(summary_tag,
                             tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                          [2, 0, 1, 3]),
                             max_images=10)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer)
        return ret + b
    else:
        return ret

def wndense(x, size, name, init_scale=1.0):
    v = tf.get_variable(name + "/V", [int(x.get_shape()[1]), size],
                        initializer=tf.random_normal_initializer(0, 0.05))
    g = tf.get_variable(name + "/g", [size], initializer=tf.constant_initializer(init_scale))
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(0.0))

    # use weight normalization (Salimans & Kingma, 2016)
    x = tf.matmul(x, v)
    scaler = g / tf.sqrt(sum(tf.square(v), axis=0, keepdims=True))
    return tf.reshape(scaler, [1, size]) * x + tf.reshape(b, [1, size])

def densenobias(x, size, name, weight_init=None):
    return dense(x, size, name, weight_init=weight_init, bias=False)

def dropout(x, pkeep, phase=None, mask=None):
    mask = tf.floor(pkeep + tf.random_uniform(tf.shape(x))) if mask is None else mask
    if phase is None:
        return mask * x
    else:
        return switch(phase, mask*x, pkeep*x)

def batchnorm(x, name, phase, updates, gamma=0.96):
    k = x.get_shape()[1]
    runningmean = tf.get_variable(name+"/mean", shape=[1, k], initializer=tf.constant_initializer(0.0), trainable=False)
    runningvar = tf.get_variable(name+"/var", shape=[1, k], initializer=tf.constant_initializer(1e-4), trainable=False)
    testy = (x - runningmean) / tf.sqrt(runningvar)

    mean_ = mean(x, axis=0, keepdims=True)
    var_ = mean(tf.square(x), axis=0, keepdims=True)
    std = tf.sqrt(var_)
    trainy = (x - mean_) / std

    updates.extend([
        tf.assign(runningmean, runningmean * gamma + mean_ * (1 - gamma)),
        tf.assign(runningvar, runningvar * gamma + var_ * (1 - gamma))
    ])

    y = switch(phase, trainy, testy)

    out = y * tf.get_variable(name+"/scaling", shape=[1, k], initializer=tf.constant_initializer(1.0), trainable=True)\
            + tf.get_variable(name+"/translation", shape=[1,k], initializer=tf.constant_initializer(0.0), trainable=True)
    return out



# ================================================================
# Basic Stuff
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *inputs : type(outputs)(zip(outputs.keys(), f(*inputs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *inputs : f(*inputs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        assert all(len(i.op.inputs)==0 for i in inputs), "inputs should all be placeholders"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(zip(self.inputs, inputvals))
        feed_dict.update(self.givens)
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results

def mem_friendly_function(nondata_inputs, data_inputs, outputs, batch_size):
    if isinstance(outputs, list):
        return _MemFriendlyFunction(nondata_inputs, data_inputs, outputs, batch_size)
    else:
        f = _MemFriendlyFunction(nondata_inputs, data_inputs, [outputs], batch_size)
        return lambda *inputs : f(*inputs)[0]

class _MemFriendlyFunction(object):
    def __init__(self, nondata_inputs, data_inputs, outputs, batch_size):
        self.nondata_inputs = nondata_inputs
        self.data_inputs = data_inputs
        self.outputs = list(outputs)
        self.batch_size = batch_size
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.nondata_inputs) + len(self.data_inputs)
        nondata_vals = inputvals[0:len(self.nondata_inputs)]
        data_vals = inputvals[len(self.nondata_inputs):]
        feed_dict = dict(zip(self.nondata_inputs, nondata_vals))
        n = data_vals[0].shape[0]
        for v in data_vals[1:]:
            assert v.shape[0] == n
        for i_start in range(0, n, self.batch_size):
            slice_vals = [v[i_start:min(i_start+self.batch_size, n)] for v in data_vals]
            for (var,val) in zip(self.data_inputs, slice_vals):
                feed_dict[var]=val
            results = tf.get_default_session().run(self.outputs, feed_dict=feed_dict)
            if i_start==0:
                sum_results = results
            else:
                for i in range(len(results)):
                    sum_results[i] = sum_results[i] + results[i]
        for i in range(len(results)):
            sum_results[i] = sum_results[i] / n
        return sum_results

# ================================================================
# Modules
# ================================================================

class Module(object):
    def __init__(self, name):
        self.name = name
        self.first_time = True
        self.scope = None
        self.cache = {}
    def __call__(self, *args):
        if args in self.cache:
            print("(%s) retrieving value from cache"%self.name)
            return self.cache[args]
        with tf.variable_scope(self.name, reuse=not self.first_time):
            scope = tf.get_variable_scope().name
            if self.first_time:
                self.scope = scope
                print("(%s) running function for the first time"%self.name)
            else:
                assert self.scope == scope, "Tried calling function with a different scope"
                print("(%s) running function on new inputs"%self.name)
            self.first_time = False
            out = self._call(*args)
        self.cache[args] = out
        return out
    def _call(self, *args):
        raise NotImplementedError

    @property
    def trainable_variables(self):
        assert self.scope is not None, "need to call module once before getting variables"
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def variables(self):
        assert self.scope is not None, "need to call module once before getting variables"
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)


def module(name):
    @functools.wraps
    def wrapper(f):
        class WrapperModule(Module):
            def _call(self, *args):
                return f(*args)
        return WrapperModule(name)
    return wrapper

# ================================================================
# Graph traversal
# ================================================================

VARIABLES = {}


def get_parents(node):
    return node.op.inputs

def topsorted(outputs):
    """
    Topological sort via non-recursive depth-first search
    """
    assert isinstance(outputs, (list,tuple))
    marks = {}
    out = []
    stack = [] #pylint: disable=W0621
    # i: node
    # jidx = number of children visited so far from that node
    # marks: state of each node, which is one of
    #   0: haven't visited
    #   1: have visited, but not done visiting children
    #   2: done visiting children
    for x in outputs:
        stack.append((x,0))
        while stack:
            (i,jidx) = stack.pop()
            if jidx == 0:
                m = marks.get(i,0)
                if m == 0:
                    marks[i] = 1
                elif m == 1:
                    raise ValueError("not a dag")
                else:
                    continue
            ps = get_parents(i)
            if jidx == len(ps):
                marks[i] = 2
                out.append(i)
            else:
                stack.append((i,jidx+1))
                j = ps[jidx]
                stack.append((j,0))
    return out


# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [numel(v)])
        for (v, grad) in zip(var_list, grads)])

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype,[total_size])
        start=0
        assigns = []
        for (shape,v) in zip(shapes,var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start+size],shape)))
            start+=size
        self.op = tf.group(*assigns)
    def __call__(self, theta):
        get_session().run(self.op, feed_dict={self.theta:theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])
    def __call__(self):
        return get_session().run(self.op)

# ================================================================
# Misc
# ================================================================


def fancy_slice_2d(X, inds0, inds1):
    """
    like numpy X[inds0, inds1]
    XXX this implementation is bad
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)


def scope_vars(scope, trainable_only):
    """
    Get variables inside a scope
    The scope can be specified as a string
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )

def lengths_to_mask(lengths_b, max_length):
    """
    Turns a vector of lengths into a boolean mask

    Args:
        lengths_b: an integer vector of lengths
        max_length: maximum length to fill the mask

    Returns:
        a boolean array of shape (batch_size, max_length)
        row[i] consists of True repeated lengths_b[i] times, followed by False
    """
    lengths_b = tf.convert_to_tensor(lengths_b)
    assert lengths_b.get_shape().ndims == 1
    mask_bt = tf.expand_dims(tf.range(max_length), 0) < tf.expand_dims(lengths_b, 1)
    return mask_bt


def in_session(f):
    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        with tf.Session():
            f(*args, **kwargs)
    return newfunc


_PLACEHOLDER_CACHE = {} # name -> (placeholder, dtype, shape)
def get_placeholder(name, dtype, shape):
    print("calling get_placeholder", name)
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1==dtype and shape1==shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out,dtype,shape)
        return out
def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]

def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])

def reset():
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    tf.reset_default_graph()
