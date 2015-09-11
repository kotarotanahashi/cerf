### MNIST

```python
from coyote.Layer import *
from coyote.Model import *
from coyote.load_data import *

# load data
datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

# make neural network
layers=[]
layers.append(FullyConnect(n_in=28 * 28, n_out=100, activation='tanh'))
layers.append(FullyConnect(n_in=100, n_out=50, activation='tanh'))
layers.append(LogisticRegression(n_in=50, n_out=10))

# compile model and train
model = Model(layers)
model.fit(train_set_x, train_set_y,validation_data=[valid_set_x,valid_set_y])
```