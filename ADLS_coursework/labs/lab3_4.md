# Lab3

#### 1. Additional metrics
For evaluating the quality of the quantization, the following metrics can be taken into considerations:
   - Latency
   - Model size
   - Weight bit size
   - Number of BitOPs
  
The reason why FLOPs is not a metric is that for each quantization to integer, no matter the specific precision, the FLOPs is 0 thus the FLOPs for all the quantizations to integer should be the same.

#### 2. Implement metrics
Implement `latency`, `accuracy`, `weight bit` and `bitops` metrics by inserting the following codes to the search rounds. 
```python
recorded_accs = []
recorded_bitwise = []
recorded_time = []
recorded_size = []

for i, config in enumerate(search_spaces):
    print("Input data width: ", end="")
    print((config['linear']['config']['data_in_width'], config['linear']['config']['data_in_frac_width']), end=" ")
    print("Weight width: ", end="")
    print((config['linear']['config']['weight_width'], config['linear']['config']['weight_frac_width']), end=": ")
    mg, _ = quantize_transform_pass(mg, config)

    ### bitwise operations
    mg_bit, total_bit = flop_count.count_bitops(mg)
    ### weight bits
    _, size = calculate_avg_bits_mg_analysis_pass(mg, {})

    j = 0
    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg, time_duration = 0, 0, 0
    accs, losses, time_durations = [], [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs

        ### latency
        start = time.time()
        preds = mg.model(xs)
        end = time.time()

        loss = torch.nn.functional.cross_entropy(preds, ys)
        ### accuracy
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        time_durations.append(end-start)

        if j > num_batchs:
            break
        j += 1

    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    time_avg = sum(time_durations) / len(time_durations)
    recorded_accs.append(acc_avg)
    recorded_bitwise.append(total_bit["total_bits"])
    recorded_size.append(size["w_overall_bit"])
    recorded_time.append(time_avg)

    print("Accuracy:", np.round(acc_avg.numpy(), 4), end="  ")
    print("BitOPs:", np.round(total_bit["total_bits"], 4), end="  ")
    print("Weights bit:", np.round(size["w_overall_bit"], 4), end="  ")
    print("Latency:", np.round(time_avg, 7))
```

   The outputs are
```
    Input data width: (16, 8) Weight width: (16, 8): Accuracy: 0.4773  BitOPs: 703744.0  Weights bit: 1920  Latency: 0.0004673
    Input data width: (16, 8) Weight width: (8, 6): Accuracy: 0.4763  BitOPs: 621824.0  Weights bit: 1120  Latency: 0.0004841
    Input data width: (16, 8) Weight width: (8, 4): Accuracy: 0.4731  BitOPs: 621824.0  Weights bit: 960  Latency: 0.0004819
    Input data width: (16, 8) Weight width: (4, 2): Accuracy: 0.4582  BitOPs: 580864.0  Weights bit: 480  Latency: 0.000506
    Input data width: (8, 6) Weight width: (16, 8): Accuracy: 0.4716  BitOPs: 616704.0  Weights bit: 1920  Latency: 0.0004481
    Input data width: (8, 6) Weight width: (8, 6): Accuracy: 0.4656  BitOPs: 575744.0  Weights bit: 1120  Latency: 0.000379
    Input data width: (8, 6) Weight width: (8, 4): Accuracy: 0.4732  BitOPs: 575744.0  Weights bit: 960  Latency: 0.0004108
    Input data width: (8, 6) Weight width: (4, 2): Accuracy: 0.4367  BitOPs: 555264.0  Weights bit: 480  Latency: 0.0003997
    Input data width: (8, 4) Weight width: (16, 8): Accuracy: 0.4681  BitOPs: 616704.0  Weights bit: 1920  Latency: 0.0003918
    Input data width: (8, 4) Weight width: (8, 6): Accuracy: 0.47  BitOPs: 575744.0  Weights bit: 1120  Latency: 0.0003983
    Input data width: (8, 4) Weight width: (8, 4): Accuracy: 0.4784  BitOPs: 575744.0  Weights bit: 960  Latency: 0.0003939
    Input data width: (8, 4) Weight width: (4, 2): Accuracy: 0.4568  BitOPs: 555264.0  Weights bit: 480  Latency: 0.0003945
    Input data width: (4, 2) Weight width: (16, 8): Accuracy: 0.4646  BitOPs: 575744.0  Weights bit: 1920  Latency: 0.000388
    Input data width: (4, 2) Weight width: (8, 6): Accuracy: 0.4702  BitOPs: 555264.0  Weights bit: 1120  Latency: 0.0003922
    Input data width: (4, 2) Weight width: (8, 4): Accuracy: 0.4661  BitOPs: 555264.0  Weights bit: 960  Latency: 0.0003978
    Input data width: (4, 2) Weight width: (4, 2): Accuracy: 0.43  BitOPs: 545024.0  Weights bit: 480  Latency: 0.0003788
```
   It can be observed that quantization effect all of the four metrics. For instance, the more the model is quantized, the more the model accuracy is influenced, while the less bit the weights take.

   In this case, the reasons that accuracy and loss serve as the same quality metric are two-fold:
   1. They are highly negatively correlated, a low loss always indicates a high accuracy.
   2. They evalute the same persepective of how great the model fits the data while other metrics like latency measures how fast the model is running and the model size measures how much memory the model takes.

#### 3. Brute-force search implementation
Implement the brute force search by adding brute force sampler in the `sampler_map` function in `optuna.py`.
```python
def sampler_map(self, name):
    match name.lower():
        ### add brute-force
        case "brute-force":
            sampler = optuna.samplers.BruteForceSampler(seed=1)
        case "random":
            sampler = optuna.samplers.RandomSampler()
        case "tpe":
            sampler = optuna.samplers.TPESampler()
        case "nsgaii":
            sampler = optuna.samplers.NSGAIISampler()
        case "nsgaiii":
            sampler = optuna.samplers.NSGAIIISampler()
        case "qmc":
            sampler = optuna.samplers.QMCSampler()
        case _:
            raise ValueError(f"Unknown sampler name: {name}")
    return sampler
```
For the `toml` config, revise the `search.strategy.setup` as follows
```
[search.strategy.setup]
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "brute-force"
# sum_scaled_metrics = true # single objective
# direction = "maximize"
sum_scaled_metrics = false # multi objective
```
Then run the following commands for brute force search
```
./ch search --config configs/examples/jsc_toy_by_type.toml --load ../mase_output/jsc-tiny_classification_jsc_2024-01-29/software/training_ckpts/best.ckpt
```

#### 4. Comparison of brute-force and tpe search
Compare brute-force search with tpe based search which is default setting in `jsc_toy_by_type.toml`. We can get the following results.
   
For tpe search, the results are
![](https://github.com/Hhy096/mase/blob/main/ADLS_coursework/graph/tpe.jpg?raw=true)
For brute search, the results are
![](https://github.com/Hhy096/mase/blob/main/ADLS_coursework/graph/brute_force.jpg?raw=true)


From the results, we can see that brute force may run a little bit longer than tpe method for each iterations while tpe runs more rounds for searching the best choice. This can be due to the limited search size because tpe should be more efficient search method than brute-force.

# Lab4

#### 1. Modify the network
It is straightforward to expand layers to double their sizes. It can be done by the following codes. The `ReLU` function can take no layer size as input parameter.
   
```python
# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(),  # 1
            nn.Linear(16, 32),  # linear seq_2
            nn.ReLU(),  # 3
            nn.Linear(32, 32),  # linear seq_4
            nn.ReLU(),  # 5
            nn.Linear(32, 5),  # linear seq_6
            nn.ReLU(),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

#### 2~4 Design search method for architectures

Problem 2, 3, 4 could be done on together. To integrate the search to `chop` flow and run it from the command line to search on the architecture level, the following procedures should be followed:
1. Define the original architecture and getter function in `/chop/models/physical/jet_substructure/__init__.py` for calling by the `toml` config.
    ```python
    ### test lab4
    class JSC_Three_Linear_Layers(nn.Module):
        def __init__(self, info):
            super(JSC_Three_Linear_Layers, self).__init__()
            self.seq_blocks = nn.Sequential(
                nn.BatchNorm1d(16),  # 0
                nn.ReLU(),  # 1
                nn.Linear(16, 16),  # linear seq_2
                nn.ReLU(),  # 3
                nn.Linear(16, 16),  # linear seq_4
                nn.ReLU(),  # 5
                nn.Linear(16, 5),  # linear seq_6
                nn.ReLU(),  # 7
            )

        def forward(self, x):
            return self.seq_blocks(x)

    ### test lab4
    def get_test_lab4(info):
        return JSC_Three_Linear_Layers(info)
    ```
2. Add initiation of the model in `PHYSICAL_MODELS` of `/chop/models/physical/__init__.py`.
    ```python
    "jsc-lab4": {
            "model": get_test_lab4,
            "info": MaseModelInfo(
                "jsc-lab4",
                model_source="physical",
                task_type="physical",
                physical_data_point_classification=True,
                is_fx_traceable=True,
            ),
        },
    ```
3. Implement `GraphSearchSpaceModelStructurePTQ` class for constructing search instance while searching through the architecture space. Implement this class under the path `/mase/machop/chop/actions/search/search_space/lab4`. In this class, define the function `redefine_linear_transform_pass` for constructing corresponding masegraph of different channel multiplier values.
   ```python
    def redefine_linear_transform_pass(mg, pass_args=None):

        main_config = pass_args
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError(f"default value must be provided.")
        i = 0
        first_linear_layer = True
        for node in mg.fx_graph.nodes:
            i += 1
            if node.name == "x":
                continue
            
            # if node name is not matched, it won't be tracked
            config = main_config.get(node.name, default)['config']
            if node.meta["mase"].parameters["common"]["mase_op"] == "linear":

                if first_linear_layer:
                    initial_in_features = mg.modules[node.target].in_features
                    last_out_features = mg.modules[node.target].out_features
                    

                ori_module = mg.modules[node.target]
                current_out_features = ori_module.out_features
                bias = ori_module.bias

                channel_multiplier = config.get("channel_multiplier", 1)
                current_out_features = current_out_features * channel_multiplier
                
                if first_linear_layer:
                    new_module = instantiate_linear(initial_in_features, current_out_features, bias)
                    first_linear_layer = False
                else:
                    new_module = instantiate_linear(last_out_features, current_out_features, bias)

                parent_name, name = get_parent_name(node.target)
                setattr(mg.modules[parent_name], name, new_module)
                last_out_features = current_out_features

        return mg, {}
   ```
4. Revise `RunnerBasicTrain` class under path `mase/machop/chop/actions/search/strategies/runners/software/train.py` for adjusting models and data for the jsc task. The revision are as follows:
    ```python
    ################################################################
    ### define the loss for lab4 
    def lab4_cls_forward(self, batch, model):
        x, y = batch[0].to(self.accelerator), batch[1].to(self.accelerator)
        logits = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.metric(logits, y)
        self.loss(loss)
        return loss
    ################################################################

    def forward(self, task: str, batch: dict, model):
        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.vision_cls_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.nlp_cls_forward(batch, model)
                case "language_modeling" | "lm":
                    loss = self.nlp_lm_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        ####################################################################
        ### lab4 modification add for jsc-tiny method
        elif self.model_info.physical_data_point_classification:
            match self.task:
                case "classfication" | "cls":
                    loss = self.lab4_cls_forward(batch, model.model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        ####################################################################
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

        return loss
    ```
    For `__call__` function, the revision should be as follows. The revision let the search method consider the accuracy and loss metric in validation set as the objective function.
    ```python
    ### evaluation
    val_dataloader = data_module.val_dataloader()

    model.model.eval()
    # print("Evaluation start")
    for i, batch in enumerate(val_dataloader):
        x, y = batch[0].to(self.accelerator), batch[1].to(self.accelerator)
        logits = model.model(x)
        criterion = nn.CrossEntropyLoss()
        self.metric(logits, y)
        loss = criterion(logits, y)
        self.loss(loss)
        if i >= num_batches - 1:
            break
    ```
5. Revise `toml` config file to adjust the revised search target, name the file as `jsc_lab4_train.toml`
6. Then the search can be done by the following command:
     ```
     ./ch search --config configs/examples/jsc_lab4_train.toml
     ```
    
The results of the commands are as follows:

![](https://github.com/Hhy096/mase/blob/main/ADLS_coursework/graph/tpe.jpg?raw=true)
