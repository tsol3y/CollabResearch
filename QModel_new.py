import numpy as np
import random
import os
from collections import deque
import strawberryfields as sf

from strawberryfields.ops import *
import tensorflow as tf 
class QModel:
    # Need to adjust theta in Classifier
    def __init__(self, session, theta, shape_in, shape_out, n_layers):
        self.x = tf.placeholder(tf.float32, shape = [shape_in,1])
        self.y = tf.placeholder(tf.float32, shape = [shape_out])
        self.para = tf.Variable(theta)
        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
        circuit = sf.Program(shape_in)

        with circuit.context as q: 
            para_i = 0
            for i in range(shape_in):
                Dgate(self.x[i][0]) | q[i]
                Dgate(self.para[para_i]) | q[i]
                para_i += 1
                Sgate(self.para[para_i]) | q[i]
                para_i += 1
                Vgate(self.para[para_i]) | q[i]
                para_i += 1

            for i in range(shape_in):
                if i < shape_in-1:
                    BSgate(self.para[para_i]) | (q[i], q[i+1])
                    BSgate() | (q[i], q[i+1])
                    para_i += 1

        results = eng.run(circuit, run_options={"eval": False})
        # self.output,_ = results.state.quad_expectation(0)
        all_outputs = [results.state.quad_expectation(i)[0] for i in range(shape_in)]
        print(all_outputs)
        norm = sum(all_outputs) + 1e-10
        self.output = sum(all_outputs[:shape_out]) / norm
        # self.output = tf.Variable(0.0)
        # self.output = tf.assign(self.output,tf.add(self.output, output))
        # self.output = tf.output
       
        self.loss = tf.losses.mean_squared_error(labels=[self.output], predictions=self.y)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = [output]))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        self.train = optimizer.minimize(self.loss)
        self.sess = session
        self.sess.run(tf.global_variables_initializer())

        # self.Shape = x_size

    def calc_grad(self, X, Y):
        # gradient = tf.placeholder(tf.float32, shape = [self.Shape,1])
        gradient = []
        grad = tf.gradients(self.loss, self.para)[0]
        # for i in range(len(X)):
        # print("train ", self.train,  " ------------------- ")
        gradient.append(self.sess.run(grad, {self.x: X[-1], self.y: Y[-1]}))
        return np.array(gradient)
        
    def fit(self,X, Y):
        # grad,_ = self.sess.run(self.train, {self.x: X_j, self.y: Y_j})
        for i in range(len(X)):
            self.sess.run(self.train, {self.x: X[i], self.y: Y[i]})
        # return tf.reshape(grad, [3,1])
    # def fit(X_j, Y_j, sess):
        # sess.run(train, {x: [X_j], y: [Y_j]})
        # return sess.run(cross_entropy, {x: [X_j], y: [Y_j]})

    def predict(self,X):
        pred = []
        for i in range(len(X)):
            pred += [self.sess.run(self.output,{self.x: X[i]})]
        print("pred: ", pred, " +++++++++++++++ ")
        return pred
        

