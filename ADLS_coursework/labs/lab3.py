import sys
import logging
import os
import numpy as np
from pathlib import Path
from pprint import pprint as pp

import torch
from torchmetrics.classification import MulticlassAccuracy

import copy
import time

os.chdir("/home/honghaoyang/mase/machop")
print("Working directory: ", end="")
print(os.getcwd())

import sys
sys.path.append("/home/honghaoyang/mase/machop")

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.tools.checkpoint_load import load_model
from chop.models import get_model_info, get_model

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

from chop.passes.graph.analysis.hhy_lab_pass import flop_count
from chop.passes.graph.analysis.quantization.calculate_avg_bits import calculate_avg_bits_mg_analysis_pass


def compute_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

CHECKPOINT_PATH = '../mase_output/jsc-tiny_classification_jsc_2024-01-29/software/training_ckpts/best.ckpt'
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained = True,
    checkpoint = '../mase_output/jsc-tiny_classification_jsc_2024-01-29/software/training_ckpts/best.ckpt')
model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

# print(search_spaces)

# grid search
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 2000
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

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

    mg_bit, total_bit = flop_count.count_bitops(mg)
    _, size = calculate_avg_bits_mg_analysis_pass(mg, {})

    j = 0
    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg, time_duration = 0, 0, 0
    accs, losses, time_durations = [], [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs

        start = time.time()
        preds = mg.model(xs)
        end = time.time()

        loss = torch.nn.functional.cross_entropy(preds, ys)
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