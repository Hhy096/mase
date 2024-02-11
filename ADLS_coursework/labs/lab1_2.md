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

The table shows the training loss and validation loss with respect to the change of epochs. When the epoch is small than 50, the training loss decreases with increasing epochs. When the epoch is larger than 150, the training loss is much smaller than validation loss, indicating an overfitting problem.

#### 3. learning rate
| epoch | batch_size | learning_rate | training loss | validation loss |
|:------:|:-----:|:-----:|:------:|:------:|
|50|256|0.1|1.229|1.197|
|50|256|0.001|0.9184|0.832|
|50|256|0.0001|0.9283|0.859|
|50|256|0.00001|1.214|1.094|
|50|256|0.000001|1.388|1.354|

The table shows the training loss and validation loss with respect to the change of learning rate. It can be obeserved that when the learning rate is too large or small, the model seems to not converge. When the learning rate is within proper range, the model performs quite well.

To sum up, for learning rate, too large learning rate will cause the model wrongly converge or diverge while too small learning rate will prolong the convergence procedure. For the epoch, larger epoch might cause overfitting problem while small epoch might cause a not-fully converged model. For batch size, the loss might increase with increasing batch size.

As for the relationship between learning rates and batch sizes,
- With smaller batch sizes, since the gradient estimates are noisier, a smaller learning rate is often used to prevent the training process from becoming too unstable.
- With larger batch sizes, it's possible to use a larger learning rate because the gradient estimates are more accurate. However, the relationship is not linear, and simply increasing the learning rate proportionally with batch size does not always yield optimal results.

### Train your own network
#### 4. Implement a new network
Implement new network called **test-hhy** with **1.4k** trainable parameters to be trained (jsc-tiny has 127 trained parameters). The newly developed model should be stated in file `machop/chop/models/physical/jet_substructure/__init__.py` with a getter function.

```python
### test lab1 
class Test(nn.Module):
    def __init__(self, info):
        super(Test, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 32),  # linear              # 2
            nn.BatchNorm1d(32),  # output_quant       # 3
            nn.ReLU(32),  # 4
            # 2nd LogicNets Layer
            nn.Linear(32, 16),  # 5
            nn.BatchNorm1d(16),  # 6
            nn.ReLU(16),  # 7
            # 2nd LogicNets Layer
            nn.Linear(16, 8),  # 5
            nn.BatchNorm1d(8),  # 6
            nn.ReLU(8),  # 7
            # 3rd LogicNets Layer
            nn.Linear(8, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)

### getter function for test lab1
def get_test(info):
    return Test(info)
```

Train the new network with hyperparameters as follows and evalute the performance in test set. The train and test commands are as follows.
```
### evaluate
./ch train test-hhy jcs --max-epochs 50 --batch-size 256 --learning-rate 0.00001

### test
./ch test test-hhy test --load ../mase_output/test-hhy_classification_jsc_2024-01-30/software/training_ckpts/best.ckpt
```

We can get the following results.

| epoch | batch_size | learning_rate | training loss | validation loss|test loss|
|:------:|:-----:|:-----:|:------:|:------:|:--:|
|50|256|0.00001|0.847|0.825|0.825|



# Lab 2

### Explanation functionalities

#### 1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ….
   
The function `report_graph_analysis_pass` generates a report for the graph analysis and prints out its computational information in a table. As for the meaning of its printed jargons:
- **placeholder:** A placeholder node represents the inputs to the graph. These are the starting points of the computational graph and typically represent the input tensors to the model.
- **get_attr:** This operation is used to fetch an attribute from a module. For instance, if you have a parameter or a buffer in a **nn.Module**, accessing this parameter/buffer in the computational graph would be represented as a get_attr node.
- **call_function:** This indicates that the node represents a call to a Python function. It could be a built-in Python function, a function from a library, or a user-defined function.
- **call_method:** This indicates that the node is calling a method of an object. For example, if you have a tensor $x$ and you call $x.view(\cdots)$, the operation would be **call_method** with the method name (like view) as the target.
- **call_module:** This means the node is invoking a module in a **nn.Module** subclass.
- **output:** This represents the outputs of the graph. There is an output node in a output graph analysis, which indicates the outputs of the computation for returning or further processing.

#### 2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?
   - **profile_statistics_analysis_pass:** The function collects the tensor-wise min-max range of specified weights and bias, and the channel-wise min-max of the specified layer nodes' input activations. It first iterates through the nodes of the specified layers and revtrives the weights and activation statistics profiles and update the graph. After the iteration, compute the min max range of the collected weights and bias data.
   - **report_node_meta_param_analysis_pass:** The function is fed with the graph output from the `profile_statistics_analysis_pass` and displays the computed statistics as a table.

#### 3. Explanation of change
After `quantize_transform_pass`, only 1 OP is changed because in the `pass_args` setting from the previous cell, the `target_weight_nodes` is clarified to be `["linear"]`, indicating that only the nodes in linear layers will be iterated and modified. For other layers such as BatchNorm1d, ReLU and output, nothing is changed.

#### 4. Traverse `mg` and `ori-mg`
Write following codes to traverse `mg` and `ori_mg`, print the `node.meta["mase"].parameters["common"]` for linear node to display the detailed information of its **precision** of weights.
   ```python
   ### traverse mg
    for node in mg.fx_graph.nodes:
        ### seq_blocks_2 is with linear op
        if node.name == "seq_blocks_2":
            print(node.meta["mase"].parameters["common"])

   ### traverse ori_mg
    for node in ori_mg.fx_graph.nodes:
        ### seq_blocks_2 is with linear op
        if node.name == "seq_blocks_2":
            print(node.meta["mase"].parameters["common"])
   ```
    We can see the following results of the codes:

    Results from traversing `mg`:
    ```
    {'mase_type': 'module_related_func', 'mase_op': 'linear', 'args': {'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'integer', 'precision': [8, 4], 'value': ...}
    ```

    Results from traversing `ori_mg`:
    ```
    {'mase_type': 'module_related_func', 'mase_op': 'linear', 'args': {'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32], 'value': ...}
    ```
    We can find that `type` and `precision` are changed from `float` to `integer` and from `[32]` to `[8,4]` respectively and all the other things remained the same, indicating the graph is updated with nodes type and precision in linear layer.

#### 5. Perform quantisation flow to network in lab1
The test model in lab1 is consisted of `linear` layers. In this case, load the model and repeat the same procedure provided in the lab2 instruction file can perform the same quantisation flow of the model.

#### 6. Verification of the quantisation
Because the quantisation procedure happens only in the forward pass, simply showing the weights of `mg` and `ori_mg` cannot state that the layers are indeed quanitsed. Therefore, inputting same input to the model represented by `mg` and `ori_mg` and observing whether there are differences between outputs can be the proof of the quantisation procedure.
   ```python
   ### load same data
   test_x = iter(data_module.val_dataloader())
   xs, ys = next(test_x)

   ### mg
   for node in mg.fx_graph.nodes:
       ### print nodes with linear op
       print(node.meta["mase"].model(xs))
       break

   ### ori_mg
   for node in ori_mg.fx_graph.nodes:
       ### print nodes with linear op
       print(node.meta["mase"].model(xs))
       break
   ```
The output are as follows:

```
tensor([[0.6366, 0.2144, 0.0000, 0.0000, 3.8191],
        [0.0000, 0.0000, 2.8312, 1.4968, 0.0000],
        [0.2014, 1.6324, 0.0000, 0.0000, 0.0608],
        [0.3965, 0.0000, 0.0000, 2.5915, 0.6947],
        [3.3978, 1.1779, 0.0000, 0.0000, 0.0000],
        [1.4019, 1.9778, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.8627, 2.4588, 1.8494],
        [0.0000, 0.0000, 2.8597, 0.5348, 0.0000]], grad_fn=<ReluBackward0>)
```
and
```
tensor([[0.1210, 0.0000, 0.0000, 0.0000, 3.6378],
        [0.0000, 0.0000, 2.1936, 1.7970, 0.0664],
        [0.2963, 1.8251, 0.0000, 0.0000, 0.0000],
        [0.9156, 0.1657, 0.0000, 1.4720, 0.8020],
        [3.5517, 1.0728, 0.0000, 0.0000, 0.0000],
        [0.4814, 1.9985, 0.4557, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.4554, 2.5201, 2.1980],
        [0.0000, 0.0000, 3.3599, 1.5023, 0.0000]], grad_fn=<ReluBackward0>)
```
There are slightly differences between each entry of the two outputs, indicating that the weights of these layers are indeed quantised.

#### 7. Perform quantisation using command line interface
First change the `toml` file for loading the pre-trained JSC network, the change are as follows
```
### change
model = "test-hhy"
load_name = "../mase_output/test-hhy_classification_jsc_2024-01-30/software/training_ckpts/best.ckpt"
```

Run the following command line to conduct pretrained JSC network
```
./ch transform --config configs/examples/jsc_toy_by_type.toml --task cls --cpu=0
```

The results are the following tables:
|Original type|OP|Total|Changed|Unchanged|
|:---:|:---:|:---:|:---:|:---:|
|BatchNorm1d|batch_norm1d|5|0|5|
|Linear|lineard|4|4|5|
|ReLU|relu|5|0|5|
|output|output|1|0|1|
|x|placeholder|1|0|1|


### Optional Task

#### FLOPs
For counting the number of FLOPs, we need to follow the following steps:
- Iterate each node in the given mase graph.
- For each node, 
  - If the data input is integer, then FLOP for this node should be $0$.
  - If the data input is float, then FLOP for this node shoud be the `computations` result of the function `calculate_modules` from `chop.passes.graph.analysis.flop_estimator.calculator`.
-  After getting the FLOP, add it to `node.meta["mase"].parameters["common"]["flop"]` as a new attribute of the node.

The codes are as follows:
```python
from chop.passes.graph.analysis.flop_estimator.calculator import calculate_modules

'''
count the number of flops of models
'''
def count_flops(graph):
    total_flop = 0
    for node in graph.fx_graph.nodes:

        if node.op == "call_module":

            ### distinguish float or integere
            data_type = node.meta["mase"].parameters["common"]["args"]["data_in_0"]['type']

            if data_type == "float":
                in_data = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
                out_data = node.meta["mase"].parameters["common"]["results"]["data_out_0"]["value"]

                flop = calculate_modules(module=node.meta["mase"].module, in_data=[in_data], out_data=[out_data])

                node.meta["mase"].parameters["common"]["flop"] = flop["computations"]
                total_flop += flop["computations"]

            elif data_type == "integer":

                flop = 0
                node.meta["mase"].parameters["common"]["flop"] = 0
                total_flop += 0
    
    return graph, {"total_flops": total_flop}
```
After implementing the pass function, apply it to `mg` and `ori_mg` to check the FLOPs, we can get that `{total: 680}` for `mg` and `{total:1320}` for `ori_mg`. This indicates that the graph (`mg`) after quantisation has the whole linear layer to be out of considerations from floating computations.

#### BitOPs
For counting the number of bit-wise operations, we need to follow the following steps:
- Iterate each node in the given mase graph.
- For each node, 
  - If the layer operation is linear, the BitOPs should be the the `computations` result of the function `calculate_modules` from `chop.passes.graph.analysis.flop_estimator.calculator` times (the input data precision times weight precision plus bias precision).
  - If the layer operation is relu, the BitOPs should be the `computations` result times the input data precision.
  - If the layer operation is batch_norm1d, the BitOPs should be the `computations` result times the square of the input data precision.
- After getting the BitOPs, add it to `node.meta["mase"].parameters["common"]["bitops"]` as a new attribute of the node.

The codes are as follows:

```python
'''
count the number of Bitops of models
'''
def count_bitops(graph):
    total = 0
    for node in graph.fx_graph.nodes:

        meta = node.meta["mase"]
        mase_op = meta.parameters["common"]["mase_op"]
        mase_type = meta.parameters["common"]["mase_type"]

        if mase_type in ["module", "module_related_func"]:
            data_in_0 = meta.parameters["common"]["args"]["data_in_0"]["value"]
            data_out_0 = meta.parameters["common"]["results"]["data_out_0"]["value"]
            precision_in = meta.parameters["common"]["args"]["data_in_0"]["precision"][0]

            count = calculate_modules(meta.module, [data_in_0], [data_out_0])

            if mase_op == "linear" or mase_op == "conv1d":    
                precision_weight = meta.parameters["common"]["args"]["weight"]["precision"][0]
                precision_bias = meta.parameters["common"]["args"]["bias"]["precision"][0]
                total += count["computations"] * (precision_in * precision_weight + max(precision_bias, precision_in))

            if mase_op == "relu":
                total += count["computations"] * precision_in

            if mase_op == "batch_norm1d":
                total += (count["computations"] * (precision_in**2))

        node.meta["mase"].parameters["common"]["bitops"] = total

    return graph, {"total_flops": total}
```

After implementing the pass function, apply it to `mg` and `ori_mg` to check the FLOPs, we can get that `{total: 575744.0}` for `mg` and `{total:1205504.0}` for `ori_mg`. This indicates that the graph (`mg`) after quantisation has much more less bit-wise operations than the original mase graph.
