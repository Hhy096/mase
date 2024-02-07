# Lab4

1. It is straightforward to expand layers to double their sizes. It can be done by the following codes. The `ReLU` function can take no layer size as input parameter.
   
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
2. Problem 2, 3, 4 could be done on together. To integrate the search to `chop` flow and run it from the command line to search on the architecture level, the following procedures should be followed:
   1. Define the original architecture in `/chop/models/physical/jet_substructure/__init__.py` for calling by the `toml` config
   2. Implement `GraphSearchSpaceModelStructurePTQ` class for constructing search instance while searching through the architecture space
   3. Revise `RunnerBasicTrain` class for adjusting models and data for the jsc task
   4. Revise `toml` config file to adjust the revised search target, name the file as `jsc_lab4_train.toml`
   5. Then the search can be done by the following command:
        ```
        ./ch search --config configs/examples/jsc_lab4_train.toml
        ```