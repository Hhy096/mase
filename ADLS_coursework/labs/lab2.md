# Lab 2

### Explanation functionalities

1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ….
   
    The function `report_graph_analysis_pass` generates a report for the graph analysis and prints out its computational information in a table. As for the meaning of its printed jargons:
    - **placeholder:** A placeholder node represents the inputs to the graph. These are the starting points of the computational graph and typically represent the input tensors to the model.
    - **get_attr:** This operation is used to fetch an attribute from a module. For instance, if you have a parameter or a buffer in a **nn.Module**, accessing this parameter/buffer in the computational graph would be represented as a get_attr node.
    - **call_function:** This indicates that the node represents a call to a Python function. It could be a built-in Python function, a function from a library, or a user-defined function.
    - **call_method:** This indicates that the node is calling a method of an object. For example, if you have a tensor $x$ and you call $x.view(\cdots)$, the operation would be **call_method** with the method name (like view) as the target.
    - **call_module:** This means the node is invoking a module in a **nn.Module** subclass.
    - **output:** This represents the outputs of the graph. There is an output node in a output graph analysis, which indicates the outputs of the computation for returning or further processing.

2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?
   - **profile_statistics_analysis_pass:** The function collects the tensor-wise min-max range of specified weights and bias, and the channel-wise min-max of the specified layer nodes' input activations. It first iterates through the nodes of the specified layers and revtrives the weights and activation statistics profiles and update the graph. After the iteration, compute the min max range of the collected weights and bias data.
   - **report_node_meta_param_analysis_pass:** The function is fed with the graph output from the `profile_statistics_analysis_pass` and displays the computed statistics as a table.

3. After `quantize_transform_pass`, only 1 OP is changed because in the `pass_args` setting from the previous cell, the `target_weight_nodes` is clarified to be `["linear"]`, indicating that only the nodes in linear layers will be iterated and modified. For other layers such as BatchNorm1d, ReLU and output, nothing is changed.

4. Write following codes to traverse `mg` and `ori_mg`, print the `node.meta["mase"].parameters["common"]` for linear node to display the detailed information of its **precision** of weights.
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

5. The test model in lab1 is consisted of `linear` layers. In this case, load the model and repeat the same procedure provided in the lab2 instruction file can perform the same quantisation flow the the model.

6. Because the quantisation procedure happens only in the forward pass, simply showing the weights of `mg` and `ori_mg` cannot state that the layers are indeed quanitsed. Therefore, inputting same input to the model represented by `mg` and `ori_mg` and observing whether there are differences can be the proof of the quantisation procedure.
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

#### Optional Task

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