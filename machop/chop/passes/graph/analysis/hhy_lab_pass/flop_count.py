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