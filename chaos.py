from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor
import numpy as np
import tensorflow as tf

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    
class ESNCell(rnn_cell_impl.RNNCell):
  """Echo State Network Cell.
      Code taken from https://github.com/m-colombo/Tensorflow-EchoStateNetwork (CN)
      
    Based on http://www.faculty.jacobs-university.de/hjaeger/pubs/EchoStatesTechRep.pdf
    Only the reservoir, the randomized recurrent layer, is modelled. The readout trainable layer
    which map reservoir output to the target output is not implemented by this cell,
    thus neither are feedback from readout to the reservoir (a quite common technique).
    Here a practical guide to use Echo State Networks:
    http://minds.jacobs-university.de/sites/default/files/uploads/papers/PracticalESN.pdf
    Since at the moment TF doesn't provide a way to compute spectral radius
    of a matrix the echo state property necessary condition `max(eig(W)) < 1` is approximated
    scaling the norm 2 of the reservoir matrix which is an upper bound of the spectral radius.
    See https://en.wikipedia.org/wiki/Matrix_norm, the section on induced norms.
  """

  def __init__(self, num_units, wr2_scale=0.7, connectivity=0.1, leaky=1.0, activation=math_ops.tanh,
               win_init=init_ops.random_normal_initializer(),
               wr_init=init_ops.random_normal_initializer(),
               bias_init=init_ops.random_normal_initializer()):
    """Initialize the Echo State Network Cell.
    Args:
      num_units: Int or 0-D Int Tensor, the number of units in the reservoir
      wr2_scale: desired norm2 of reservoir weight matrix.
        `wr2_scale < 1` is a sufficient condition for echo state property.
      connectivity: connection probability between two reservoir units
      leaky: leaky parameter
      activation: activation function
      win_init: initializer for input weights
      wr_init: used to initialize reservoir weights before applying connectivity mask and scaling
      bias_init: initializer for biases
    """
    self._num_units = num_units
    self._leaky = leaky
    self._activation = activation

    def _wr_initializer(shape, dtype, partition_info=None):
      wr = wr_init(shape, dtype=dtype)

      connectivity_mask = math_ops.cast(
          math_ops.less_equal(
            random_ops.random_uniform(shape),
            connectivity),
        dtype)

      wr = math_ops.multiply(wr, connectivity_mask)

      wr_norm2 = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(wr)))

      is_norm_0 = math_ops.cast(math_ops.equal(wr_norm2, 0), dtype)

      wr = wr * wr2_scale / (wr_norm2 + 1 * is_norm_0)

      return wr

    self._win_initializer = win_init
    self._bias_initializer = bias_init
    self._wr_initializer = _wr_initializer

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """ Run one step of ESN Cell
        Args:
          inputs: `2-D Tensor` with shape `[batch_size x input_size]`.
          state: `2-D Tensor` with shape `[batch_size x self.state_size]`.
          scope: VariableScope for the created subgraph; defaults to class `ESNCell`.
        Returns:
          A tuple `(output, new_state)`, computed as
          `output = new_state = (1 - leaky) * state + leaky * activation(Win * input + Wr * state + B)`.
        Raises:
          ValueError: if `inputs` or `state` tensor size mismatch the previously provided dimension.
          """

    inputs = convert_to_tensor(inputs)
    input_size = inputs.get_shape().as_list()[1]
    dtype = inputs.dtype

    with vs.variable_scope(scope or type(self).__name__):  # "ESNCell"

      win = vs.get_variable("InputMatrix", [input_size, self._num_units], dtype=dtype,
                            trainable=False, initializer=self._win_initializer)
      wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=dtype,
                           trainable=False, initializer=self._wr_initializer)
      b = vs.get_variable("Bias", [self._num_units], dtype=dtype, trainable=False, initializer=self._bias_initializer)
    
      tf.get_variable_scope().reuse_variables()
      in_mat = array_ops.concat([inputs, state], axis=1)
      weights_mat = array_ops.concat([win, wr], axis=0)

      output = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat) + b)

    return output, output

def esn_model_fn(features, labels, mode):
    """Add proper input and output layers to Colombo's ECNCell, add training details.
    """
    with tf.name_scope("inputs"):
        input_layer= tf.transpose(tf.reshape(features["residuals"],[-1, 11, 449]), [0,2,1])
        variable_summaries(input_layer)
    
    with tf.name_scope("reservoir"):
        cell = ESNCell(num_units=1000, connectivity=0.2, wr2_scale=0.7, leaky=1.0)
        (reservoir_layer, _) = tf.nn.dynamic_rnn(cell=cell, inputs=input_layer, dtype=tf.float64)
        variable_summaries(reservoir_layer)
    
    with tf.name_scope("output"):
        output_layer = tf.layers.dense(reservoir_layer, units=11, use_bias=False, 
                                     name="residual_hat", activation=None)
        variable_summaries(output_layer)
        
    predictions={"residuals":tf.reshape(output_layer, [449, 11])}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(tf.squeeze(input_layer,0), predictions['residuals'])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, 
                                              predictions=predictions)
    
    eval_metric_ops = {'MSE':loss}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
def esn_dynamic():
    sess = tf.InteractiveSession()
    with tf.gfile.FastGFile("graphname.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
    # All operations
    sess.graph.get_operations()
    
    #Read in y_hats
    import pandas as pd
    y_hat = pd.read_csv("y_hat.csv")
    endos = pd.read_csv("data/endo.csv")
    p_input = endos.merge(y_hat, left_on=['Date','asset'], right_on=['Date','Asset'])
    residuals = (p_input.y_hat - p_input.endo).dropna()
    chaos_filter = tf.estimator.Estimator(model_fn=esn_model_fn, model_dir="~/chaos_model")
    #tensors_to_log = {"residuals":"residual_hat"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    #merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter('~/summaries_dir/train',
                                          sess.graph)
    tf.global_variables_initializer().run()
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'residuals':residuals},
        y=residuals,
        num_epochs=10,
        shuffle=False
    )
        
    ans = chaos_filter.train(input_fn=train_input_fn, steps=2000)#, hooks=[logging_hook])

    np.savetxt("chaos.csv",ans['predictions'],delimiter=",")
        
def main():
    esn_dynamic()