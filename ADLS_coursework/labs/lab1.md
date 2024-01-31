# Lab 1 for ADLS

### Varing parameters

To test the influences of different hyper parameters, run the following experiments and get the losses in training and validation sets as in the following tables:

#### 1. batch size
| epoch | batch_size | learning_rate | training loss | validation loss |
|:------:|:-----:|:-----:|:------:|:------:|
|50|64|0.0001|0.7837|0.878|
|50|128|0.0001|0.9908|0.8581|
|50|256|0.0001|0.9283|0.859|
|50|512|0.0001|1.075|1.053|
|50|1024|0.0001|1.049|1.086|

The table show the training loss and validation loss with respect to the change of batch size. It seems that the loss tends to increase with increasing batch size.

#### 2. max epoch

| epoch | batch_size | learning_rate | training loss | validation loss |
|:------:|:-----:|:-----:|:------:|:------:|
|10|256|0.0001|0.9919|0.9922|
|50|256|0.0001|0.9283|0.859|
|100|256|0.0001|1.073|0.8435|
|150|256|0.0001|0.8844|0.84|
|200|256|0.0001|0.6919|0.8382|

The table show the training loss and validation loss with respect to the change of epochs. When the epoch is small than 50, the training loss decreases with increasing epochs. When the epoch is larger than 150, the training loss is much smaller than validation loss, indicating an overfitting problem.

#### 3. learning rate
| epoch | batch_size | learning_rate | training loss | validation loss |
|:------:|:-----:|:-----:|:------:|:------:|
|50|256|0.1|1.229|1.197|
|50|256|0.001|0.9184|0.832|
|50|256|0.0001|0.9283|0.859|
|50|256|0.00001|1.214|1.094|
|50|256|0.000001|1.388|1.354|

The table show the training loss and validation loss with respect to the change of learning rate. It can be obeserved that when the learning rate is too large or small, the model seems to not converge. When the learning rate is within proper range, the model performs quite well.

To sum up, for learning rate, too large learning rate will cause the model wrongly converge or diverge while too small learning rate will prolong the convergence procedure. For the epoch, larger epoch might cause overfitting problem while small epoch might cause a not-fully converged model. For batch size, the loss might increase with increasing batch size.

### Train your own network

Implement new network called **test-hhy** with **1.4k** trainable parameters to be trained (jsc-tiny has 127 trained parameters). Train the new network with hyperparameters as follows and evalute the performance in test set. We can get the following results.

| epoch | batch_size | learning_rate | training loss | validation loss|test loss|
|:------:|:-----:|:-----:|:------:|:------:|:--:|
|50|256|0.00001|0.847|0.825|0.825|