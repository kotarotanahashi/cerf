## Usage of Model

Activations can either be used through an `Activation` layer, or through the `activation` argument supported by all forward layers:

⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅

```python
from keras.layers.core import Activation, Dense

model.add(Dense(64, 64, init='uniform'))
model.add(Activation('tanh'))
```
is equivalent to:
```python
model.add(Dense(20, 64, init='uniform', activation='tanh'))
```
