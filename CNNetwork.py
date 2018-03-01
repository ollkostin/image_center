import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import numpy


class CNNetwork(object):
    def __init__(self, rng, input, filter_shape, image_shape, pool_size=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(pool_size))

        w_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(
                low=-w_bound,
                high=w_bound,
                size=filter_shape),
            dtype=theano.config.floatX),
            borrow=True)

        biases = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=biases, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape)

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input
