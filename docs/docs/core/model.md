### Usage of Model

`Model()` class creates an instance of model from the list of layer (instance of coyote.layer)
```python
model = Model(layers)
model.fit(X, Y, validation_data=[valid_set_x,valid_set_y])
```
###methods
* `fit(self,x,y,batch_size=600,n_epochs=100,validation_data=none)`
    * argments
        * **x**: data
        * **y**: teacher label
        * **batch_size**: the number of data used for a batch process. *default*:600
        * **n_epochs**: the number of loops to train models.
        * **validation_data**: [valid_set_x,valid_set_y] list of validation data and label
        
