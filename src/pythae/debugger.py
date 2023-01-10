import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.utils.data as data
import math 
import torch.nn as nn
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import tqdm
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export

class Debugger:
    """Base class to perform model debugging.

    """

    def __init__(
        self,
        training_dir,
        train_set, 
        device,
    ):

        self.writer = SummaryWriter(log_dir=training_dir)
        self.train_set = train_set
        self.device = device


    def tensorboard_log(self, model, model_output, optimizer, global_step, global_norm=None, losses=None, mode='train'):
        
        if losses is not None:
            for key, value in losses.items():
                self.writer.add_scalar(f"Losses/{key}", value, global_step)

        #self.writer.add_histogram("Distributions/target", targets, global_step, bins=20)
        #self.writer.add_histogram("Distributions/output", torch.clamp(outputs, min=-1., max=1.), global_step, bins=20)

       

        self.writer.add_histogram('OutputLayer/means', model_output.mu, global_step, bins=30)
        self.writer.add_histogram('OutputLayer/log_scales', model_output.std, global_step, bins=30)

        self.visualize_activations(model, global_step)
        self.visualize_gradients(model, global_step)
        self.measure_number_dead_neurons(model)

        self.writer.flush()

    def visualize_activations(self, net, global_step, color="C0"):
        activations = {}

        net.eval()
        with torch.no_grad():
            for inputs in self.train_set:
                inputs = self._set_inputs_to_device(inputs)['data'].unsqueeze(0)
                break

            # We need to manually loop through the layers to save all activations
            for layer_index, layer in enumerate(net.encoder.layers):
                inputs = layer(inputs)
                if isinstance(net.encoder.layers[layer_index][1], nn.Linear):
                    activations[layer_index] = inputs.view(-1).cpu().numpy()
            layer_index += 1
            inputs_z = net.encoder.embedding_layer(inputs)
            activations[layer_index] = inputs_z.view(-1).cpu().numpy()
            layer_index += 1
            inputs_std = net.encoder.log_var_layer(inputs)
            activations[layer_index] = inputs_std.view(-1).cpu().numpy()
         
        for key in activations:
            if key==4:
                self.writer.add_histogram('Activations - Layer 4 - Linear', activations[key], global_step)
            if key==5:
                self.writer.add_histogram('Activations - Layer mu - Linear', activations[key], global_step)
            if key==6:
                self.writer.add_histogram('Activations - Layer std - Linear', activations[key], global_step)
          

    def visualize_gradients(self, net, global_step, color="C0"):
        """
        Inputs:
            net - Object of class BaseNetwork
            color - Color in which we want to visualize the histogram (for easier separation of activation functions)
        """
        net.eval()
        for inputs in self.train_set:
            inputs = self._set_inputs_to_device(inputs)
            inputs['data'] = inputs['data'].unsqueeze(0)
            break

        # Pass one batch through the network, and calculate the gradients for the weights
        net.zero_grad()
        model_output = net(inputs, epoch=global_step, dataset_size=len(self.train_set))
       
        loss = model_output.loss
        loss.backward()
        # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
        grads = {name: params.grad.data.view(-1).cpu().clone().numpy() for name, params in net.named_parameters() if "weight" in name}
        net.zero_grad()

        ## Plotting
        self.writer.add_histogram('Gradients - Layer 4 - Linear', grads['encoder.layers.4.1.weight'], global_step)
        self.writer.add_histogram('Gradients - Layer mu - Linear', grads['encoder.embedding_layer.weight'], global_step)
        self.writer.add_histogram('Gradients - Layer std - Linear', grads['encoder.log_var_layer.weight'], global_step)
          
      


    def measure_number_dead_neurons(self, net):
    
        # For each neuron, we create a boolean variable initially set to 1. If it has an activation unequals 0 at any time,
        # we set this variable to 0. After running through the whole training set, only dead neurons will have a 1.
        neurons_dead = [
            torch.ones(layer.weight.shape[0], device=self.device, dtype=torch.bool) for layer in [net.encoder.layers[4][1], net.encoder.embedding_layer, net.encoder.log_var_layer]
        ] # Same shapes as hidden size in BaseNetwork
    
        net.eval()
        with torch.no_grad():
          
            for inputs in self.train_set:
                layer_index = 0
                inputs = self._set_inputs_to_device(inputs)['data'].unsqueeze(0)
                
                for i in range(5):
                    inputs = net.encoder.layers[i](inputs)
                    if isinstance(net.encoder.layers[i][1], nn.Linear):
                        neurons_dead[layer_index] = torch.logical_and(neurons_dead[layer_index], (inputs == 0).all(dim=0))
                        layer_index += 1
            
                inputs_z = net.encoder.embedding_layer(inputs)
                neurons_dead[layer_index] = torch.logical_and(neurons_dead[layer_index], (inputs_z == 0).all(dim=0))
                layer_index += 1
                inputs_std = net.encoder.log_var_layer(inputs)
                neurons_dead[layer_index] = torch.logical_and(neurons_dead[layer_index], (inputs_std == 0).all(dim=0))
                layer_index += 1
                        
        number_neurons_dead = [t.sum().item() for t in neurons_dead]
        print("Number of dead neurons:", number_neurons_dead)
        print("In percentage:", ", ".join([f"{(100.0 * num_dead / tens.shape[0]):4.2f}%" for tens, num_dead in zip(neurons_dead, number_neurons_dead)]))
     



    def _set_inputs_to_device(self, inputs: Dict[str, Any]):
  
        cuda_inputs = dict.fromkeys(inputs)
        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].to(self.device)
            else:
                cuda_inputs = inputs[key]
    
        inputs_on_device = cuda_inputs

        return inputs_on_device
