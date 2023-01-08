import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

class Debugger:
    """Base class to perform model debugging.

    """

    def __init__(
        self,
        training_dir,
    ):

        self.writer = SummaryWriter(log_dir=training_dir)


    def tensorboard_log(self, model, model_output, optimizer, global_step, global_norm=None, losses=None, mode='train'):
        
        if losses is not None:
            for key, value in losses.items():
                self.writer.add_scalar(f"Losses/{key}", value, global_step)

        #self.writer.add_histogram("Distributions/target", targets, global_step, bins=20)
        #self.writer.add_histogram("Distributions/output", torch.clamp(outputs, min=-1., max=1.), global_step, bins=20)

       

        self.writer.add_histogram('OutputLayer/means', model_output.mu, global_step, bins=30)
        self.writer.add_histogram('OutputLayer/log_scales', model_output.std, global_step, bins=30)

        if mode == 'train':
            for variable in model.parameters():
                self.writer.add_histogram("Weights/{}".format(variable.name), variable, global_step)

            # Get the learning rate from the optimizer
            self.writer.add_scalar("Schedules/learning_rate", optimizer.param_groups[0]['lr'],
                              global_step)

            #if updates is not None:
            #    for layer, update in updates.items():
            #        self.writer.add_scalar("Updates/{}".format(layer), update, global_step)
#
            #    max_updates = torch.max(torch.stack(list(updates.values())))
            #    assert global_norm is not None
            #    self.writer.add_scalar("Mean_Max_Updates/Global_norm", global_norm, global_step)
            #    self.writer.add_scalar("Mean_Max_Updates/Max_updates", max_updates, global_step)

        self.writer.flush()
