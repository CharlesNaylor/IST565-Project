import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.ops import math_ops
from ESN import EchoStateRNNCell
import pickle

def NRMSE(P, Y):
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(P, Y))) / (tf.reduce_max(Y) - tf.reduce_min(Y))

def cross(a, b=None):
    """Weirdly, tensorflow cross requires the inner dimension be 3!?"""
    if b is None:
        b = a
    return tf.matmul(tf.transpose(a),b)

def filter_chaos(residuals, num_units=10, batches=1,
                 stime=None, num_inputs=None, out_function=None):
    if stime is None:
        stime = residuals.shape[0]
    if num_inputs is None:
        num_inputs = residuals.shape[1]
    if out_function is None:
        out_function = lambda x: math_ops.tanh(x)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    random_seed=8675309

    # input
    rnn_inputs = np.split(residuals, batches)
    rnn_init_state = np.zeros([batches, num_units], dtype="float32")

    # output target
    rnn_target = np.split(residuals.shift(-1), batches)
    tf.reset_default_graph()

    #setup graph
    graph = tf.Graph()
    with graph.as_default() as g:
        rng = np.random.RandomState(random_seed)
        lr = 0.01
        # Build the graph
        inputs = tf.placeholder(tf.float32, [batches, stime, num_inputs])
        target = tf.placeholder(tf.float32, [stime, num_inputs])
        init_state = tf.placeholder(tf.float32, [1, num_units])
        # Init the ESN cell
        print("Making ESN init graph ...")
        cell = EchoStateRNNCell(num_units=num_units,
                                activation=out_function,
                                decay=0.1,
                                alpha=0.5,
                                rng=rng,
                                optimize=True,
                                optimize_vars=["rho", "decay","alpha", "sw"])
        print("Done")
        # cell spreading of activations
        print("Making ESN spreading graph ...")
        states = []
        state = init_state
        for t in range(stime):
            state,_ = cell(inputs=inputs[0,t:(t+1),:], state=state)
            states.append(state)
        outputs = tf.reshape(states, [stime, num_inputs, num_units])
        print("Done")

        # collapse to multivariate normal
        ro_layer = tf.squeeze(tf.layers.dense(outputs, 1))

        print("Making regression graph ...")
        # do the regression on a training subset of the timeseries
        begin = 0
        end = 104
        # optimize lambda, too
        lmb = tf.get_variable("lmb", initializer=0.1,
                              dtype=tf.float32, trainable=True)
        ro_slice = ro_layer[begin:end,:]

        # calc weighted out
        Wout = tf.matmul(
                        tf.matrix_inverse(cross(ro_slice)
                            + lmb+tf.eye(num_inputs)),
                        cross(ro_slice, target[begin:end,:]) )
        print("Done")

        # readout
        print("Making readout spreading graph ...")
        readouts = tf.matmul(ro_layer, Wout)
        print("Done")

        # train graph
        print("Making training graph ...")
        # calculate the loss over all the timeseries (escluded the beginning)
        clip_rho = cell.rho.assign(tf.clip_by_value(cell.rho, 0.0, 1.0))
        clip_alpha = cell.alpha.assign(tf.clip_by_value(cell.alpha, 0.0, 1.0))
        clip_decay = cell.decay.assign(tf.clip_by_value(cell.decay, 0.0, 1.0))
        clip_sw = cell.decay.assign(tf.clip_by_value(cell.sw, 0.0001, 0.5))
        clip_lmb = cell.decay.assign(tf.clip_by_value(lmb, 0.0001, 0.5))
        clip = tf.group(clip_rho, clip_alpha, clip_decay,clip_sw, clip_lmb)
        loss = NRMSE(target[begin:end,:], readouts[begin:end,:])
        try: # if optimize == True
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train = optimizer.minimize(loss)
        except ValueError: # if optimize == False
            train = tf.get_variable("trial", (), dtype=None)

    # run session
    from tensorflow.python import debug as tf_debug
    with graph.as_default() as g:
        trials = 2000
        with tf.Session(config=config) as session:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.run(tf.global_variables_initializer())
            losses = np.zeros(trials)
            print("Executing the graph")
            for k in range(trials):
                rho, alpha, decay, sw, U, curr_outputs, curr_readouts,curr_loss,_ = \
                        session.run([cell.rho, cell.alpha, cell.decay, cell.sw,
                                    cell.U, outputs, readouts,
                                    loss, train ],
                                    feed_dict={
                                        inputs:rnn_inputs,
                                        target: rnn_target,
                                        init_state:rnn_init_state})
                session.run(clip)
                if k%20 == 0 or k == trials-1:
                    sys.stdout.write("step: {:4d}\t".format(k))
                    sys.stdout.write("NRMSE: {:5.3f}\t".format(curr_loss))
                    sys.stdout.write("rho: {:5.3f}\t".format(rho))
                    sys.stdout.write("alpha: {:5.3f}\t".format(alpha))
                    sys.stdout.write("decay: {:5.3f}\t".format(decay))
                    sys.stdout.write("sw: {:5.3f}\n".format(decay))

                losses[k] = curr_loss
            print("Done")
    return dict(rho=rho, alpha=alpha, decay=decay, sw=sw, U=U, curr_outputs=curr_outputs,
                curr_readouts=curr_readouts, curr_loss=curr_loss)

def main():
    y_hat = pd.read_csv("y_hat.csv")
    endos = pd.read_csv("data/endo.csv").melt(id_vars="Date", var_name='Asset', value_name='endo')
    p_input = endos.merge(y_hat)
    p_input['residuals'] = (p_input.y_hat - p_input.endo).dropna()
    residuals = p_input.pivot(index='Date', columns='Asset', values='residuals')
    ans = filter_chaos(residuals, num_units=10)
    np.savetxt("residuals_hat.csv",ans['curr_readouts'],delimiter=",")
    with open("chaos.pkl", "rb") as f:
        pickle.dump(ans, f)

if __name__ == "__main__":
    sys.exit(main())
