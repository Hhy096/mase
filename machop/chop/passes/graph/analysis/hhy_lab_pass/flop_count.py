import logging
import math

import toml
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from chop.passes.graph.analysis.utils import (
    is_tensor_constant,
    match_and_filter,
    is_seq_blocks_parameter,
    get_input_nodes,
    get_output_nodes,
)
from chop.passes.graph.common import (
    MASE_BUILTIN_FUNCS,
    MASE_IMPLICIT_FUNCS,
    MASE_MODULE_RELATED_FUNCS,
)
from chop.ir.graph.mase_metadata import MaseMetadata
from chop.passes.graph.analysis.utils import fetch_attr, load_arg
from tabulate import tabulate
from torch import nn

logger = logging.getLogger(__name__)

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
    
    return graph, {"total_bits": total}
