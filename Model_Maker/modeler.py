### modeler ###
## generates models based on input ##
# for automated model construction #

from keras.models import Model, Input
from keras.layers import Concatenate


# use do to instantiate layers #
# allows models to be procedurally generated #
def do(func,*arg,**args):
    return func(*arg,**args)


# use do2 models to recursively build models #
# allows construction of models, and models made out of models
# sample tescode #
def do2(array):
    return do(array[0],array[1],**array[2])

# given a set of layers or models returns a stacked model
def make(layers,input_shape):
    input = Input(**input_shape)
    if layers[0][1] is None and layers[0][2] is None: layer = layers[0][0](input)
    else: layer = do(layers[0][0],layers[0][1],**layers[0][2])(input)
    if len(layers)>1:
        for i in range(1,len(layers)):
            if layers[i][1] is None and layers[i][2] is None: layer = layers[i][0](layer)
            else: layer = do(layers[i][0],layers[i][1],**layers[i][2])(layer)
    return Model(inputs=input, outputs=layer)

# given a set of layers returns a deep net
def make_deepnet(layers, shape, nodes, depth):
    # print (shape)
    _input = Input(shape=(shape,))
    _stacks = [_input]
    _concat = _input
    for i in range(depth):
        for n in range(len(layers)):
            if n==0:
                _layer = do(layers[n][0],layers[n][1],**layers[n][2])(_concat)
            else :
                _layer = do(layers[n][0], layers[n][1], **layers[n][2])(_layer)
        _stacks.append(_layer)
        _concat = Concatenate()(_stacks)
    return Model(inputs=_input, outputs=_layer)

def make_LSTMdeepnet(layers, shape, nodes, depth):
    # print (shape)
    _input = Input(shape=shape)
    _stacks = [_input]
    _concat = _input
    for i in range(depth):
        for n in range(len(layers)):
            if n==0:
                _layer = do(layers[n][0],layers[n][1],**layers[n][2])(_concat)
            else :
                _layer = do(layers[n][0], layers[n][1], **layers[n][2])(_layer)
        _stacks.append(_layer)
        _concat = Concatenate()(_stacks)
    return Model(inputs=_input, outputs=_layer)

def make_stateLSTMdeepnet(layers, shape, nodes, depth):
    # print (shape)
    _input = Input(batch_shape=shape)
    _stacks = [_input]
    _concat = _input
    for i in range(depth):
        for n in range(len(layers)):
            if n==0:
                _layer = do(layers[n][0],layers[n][1],**layers[n][2])(_concat)
            else :
                _layer = do(layers[n][0], layers[n][1], **layers[n][2])(_layer)
        _stacks.append(_layer)
        _concat = Concatenate()(_stacks)
    return Model(inputs=_input, outputs=_layer)

# testing implementation #
'''
import numpy as np
from keras.layers import Dense
from keras import optimizers
data = np.random.randint(-100,100,size=(300,2))/100
layers = [[Dense,(20),{'activation':'linear'}],
          [Dense,(2),{'activation':'linear'}]]
# model = make(layers,{'shape':(data.shape[-1],)})
model1 = make([layers[0]],{'shape':(data.shape[-1],)})
model2 = make([layers[1]],{'shape':model1.output_shape})
model3 = make([[model1,None,None],[model2,None,None]],{'shape':(data.shape[-1],)})
model3.compile(optimizer=optimizers.adam(lr=1e-3, beta_2=.98, epsilon=1e-5),loss='mean_squared_error')
model3.fit(data,data,epochs=50,verbose=2,batch_size=10)
'''