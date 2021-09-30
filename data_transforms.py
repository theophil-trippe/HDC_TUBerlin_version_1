import torch


# ----- data transforms -----
class Permute(object):
    """ Permute order of input tuples. """

    def __init__(self, perm):
        self.perm = perm

    def __call__(self, inputs):
        out = tuple([inputs[k] for k in self.perm])
        return out

class Shift(object):
    """ Shift. """

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, inputs):
        return inputs - self.shift

class Avg_Pool(object):
    """ Pool. """

    def __init__(self, pooling_size=2):
        self.pooling_size = pooling_size

    def __call__(self, inputs):
        return torch.nn.AvgPool2d(kernel_size=self.pooling_size, stride=self.pooling_size)(inputs)

class Apply_Fct_To_Input(object):
    def __init__(self, fct):
        self.fct = fct

    def __call__(self, inputs):
        return self.fct(inputs.unsqueeze(0)).squeeze(0)
