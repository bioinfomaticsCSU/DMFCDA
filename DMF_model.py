import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from  hyperparams import Hyperparams as params

learning_rate = params.learning_rate
threshold = params.threshold

L_layer1_num = params.L_layer1_num
L_layer2_num = params.L_layer2_num
L_layer3_num = params.L_layer3_num

R_layer1_num = params.R_layer1_num
R_layer2_num = params.R_layer2_num
R_layer3_num = params.R_layer3_num

keep_prob = params.keep_prob
reg = params.reg

class DMF(object):
    def __init__(self, dataType):
        self.dataType = dataType
        if self.dataType == 1:
            col_num = params.col_num1
            row_num = params.row_num1
        if self.dataType == 2:
            col_num = params.col_num2
            row_num = params.row_num2

        with tf.name_scope('input'):
            self.Y_input = tf.placeholder(tf.int32, [None, 1])
            self.XL_input = tf.placeholder(tf.float32, [None, col_num])
            self.XR_input = tf.placeholder(tf.float32, [None, row_num])
        with tf.variable_scope('DMF_net'):
            with tf.name_scope('L_net'):
                XL_emb1 = layers.fully_connected(self.XL_input, L_layer1_num,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 activation_fn=tf.nn.relu,
                                                 weights_regularizer=layers.l2_regularizer(scale=reg))
                XL_emb1 = tf.nn.dropout(XL_emb1, keep_prob=keep_prob)
                XL_emb2 = layers.fully_connected(XL_emb1, L_layer2_num,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 activation_fn=tf.nn.relu,
                                                 weights_regularizer=layers.l2_regularizer(scale=reg))
                XL_emb2 = tf.nn.dropout(XL_emb2, keep_prob=keep_prob)
                XL_emb3 = layers.fully_connected(XL_emb2, L_layer3_num,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 activation_fn=tf.nn.relu,
                                                 weights_regularizer=layers.l2_regularizer(scale=reg))
                XL_emb3 = tf.nn.dropout(XL_emb3, keep_prob=keep_prob)

            with tf.name_scope('R_net'):
                XR_emb1 = layers.fully_connected(self.XR_input, R_layer1_num,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 activation_fn=tf.nn.relu,
                                                 weights_regularizer=layers.l2_regularizer(scale=reg))
                XR_emb1 = tf.nn.dropout(XR_emb1, keep_prob=keep_prob)
                XR_emb2 = layers.fully_connected(XR_emb1, R_layer2_num,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 activation_fn=tf.nn.relu,
                                                 weights_regularizer=layers.l2_regularizer(scale=reg))
                XR_emb2 = tf.nn.dropout(XR_emb2, keep_prob=keep_prob)
                XR_emb3 = layers.fully_connected(XR_emb2, R_layer3_num,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 activation_fn=tf.nn.relu,
                                                 weights_regularizer=layers.l2_regularizer(scale=reg))
                XR_emb3 = tf.nn.dropout(XR_emb3, keep_prob=keep_prob)

            with tf.name_scope('Latent_concate'):
                fuse_tensor = tf.concat([XL_emb3, XR_emb3], axis=1)
                # fuse_tensor = tf.multiply(XL_emb3, XR_emb3)

            with tf.name_scope('Penult_fullyConnected'):
                penultOutput = layers.fully_connected(fuse_tensor, 16,
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                     biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                     activation_fn=tf.nn.relu,
                                                     weights_regularizer=layers.l2_regularizer(scale=reg))

            with tf.name_scope('Output'):
                logits = layers.fully_connected(penultOutput, 1,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                activation_fn=None,
                                                weights_regularizer=layers.l2_regularizer(scale=reg))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.Y_input, tf.float32), logits=logits))
        with tf.name_scope('Train'):
            reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'DMF_net')
            self.reg_loss = tf.reduce_mean(reg_ws)
            self.score = tf.nn.sigmoid(logits)
            self.loss = tf.reduce_mean(loss) + self.reg_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            self.prediction = tf.cast(tf.greater_equal(self.score, threshold), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.Y_input), tf.float32))
