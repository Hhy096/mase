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

5. 