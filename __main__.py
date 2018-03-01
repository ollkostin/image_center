# coding=utf-8

import theano
from theano import tensor as T
from theano.tensor.tests.mlp_test import HiddenLayer

from CNNetwork import *
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import numpy
import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
# [int] labels

print('... building the model')

layer0_input = x.reshape((batch_size, 1, 28, 28))

layer0 = CNNetwork(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 28, 28),
    filter_shape=(nkerns[0], 1, 5, 5),
    pool_size=(2, 2)
)

layer1 = CNNetwork(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 12, 12),
    filter_shape=(nkerns[1], nkerns[0], 5, 5),
    pool_size=(2, 2)
)

layer2_input = layer1.output.flatten(2)

layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 4 * 4,
    n_out=500,
    activation=T.tanh
)

layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

cost = layer3.negative_log_likelihood(y)

test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

params = layer3.params + layer2.params + layer1.params + layer0.params

grads = T.grad(cost, params)

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }

)
# rng = numpy.random.RandomState(23455)
# input = T.tensor4(name='input')
# 
# # weights
# w_shape = (2, 3, 9, 9)
# w_bound = numpy.sqrt(3 * 9 * 9)
# W = theano.shared(numpy.asarray(
#     rng.uniform(
#         low=-1 / w_bound,
#         high=1 / w_bound,
#         size=w_shape),
#     dtype=input.dtype), name='W')
# 
# b_shape = (2,)
# b = theano.shared(numpy.asarray(
#     rng.uniform(
#         low=- .5,
#         high=.5,
#         size=b_shape),
#     dtype=input.dtype), name='b')
# 
# conv_out = conv2d(input, W)
# 
# pooled_out = pool.pool_2d(
#     input=conv_out,
#     ds=(2, 2),
#     ignore_border=False
# )
# 
# output = T.nnet.sigmoid(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
# 
# f = theano.function(inputs=[input], outputs=output)
#
# img = Image.open('pic.png', 'r').getdata()
# img = numpy.fromstring(image.tostring(), dtype='uint8', count=-1, sep='')
# .reshape(image.shape + (len(image.getbands()),))
# img_width, img_height = (img.shape[0], img.shape[1])
# 
# img_transposed = img.transpose((2, 0, 1)).reshape(1, 3, img_height, img_width)
# 
# print img.shape
# 
# filtered_img = f(img_transposed)
# 
# pylab.subplot(1, 3, 1)
# pylab.axis('off')
# pylab.imshow(img)
# pylab.subplot(1, 3, 2)
# pylab.axis('off')
# pylab.imshow(filtered_img[0, 0, :, :])
# pylab.subplot(1, 3, 3)
# pylab.axis('off')
# pylab.imshow(filtered_img[0, 1, :, :])
# pylab.show()
