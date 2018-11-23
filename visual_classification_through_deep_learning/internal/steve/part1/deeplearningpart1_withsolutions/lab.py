# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:50:13 2018

@author: GuoOu
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from render import *

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print ("Tests passed!!")

def step(x):
  return x > 0
  
def sigmoid(x, slope_at_zero=1):
  return 1. / (1 + np.e ** (-4 * slope_at_zero * x))
  
def accuracy(desired_out, out):
  return -.5 * (desired_out - out) ** 2

class Neuron:
  def __init__(self, name, input_weight_pairs):
    self.name = str(name)
    self.weight_dict = {input:weight for (input, weight) in input_weight_pairs}
    self.inputs = [input for (input, weight) in self.weight_dict.items()]
    self.neuron_inputs = [input for input in self.inputs if input == str(input)]
  
  # Recalculate neuron's output based on input neurons most recent outputs and threshold function
  def call(self, output_dict, threshold_function):
    output_dict[self.name] = threshold_function(sum([output_dict.get(input, input) * weight for (input, weight) in self.weight_dict.items()]))
    
  # Retrieve weight for a given input neuron
  def get_weight(self, input):
    return self.weight_dict[input]
  
  # Update weight for a given input neuron
  def update_weight(self, input, new_weight):
    self.weight_dict[input] = new_weight

class NN:
  def __init__(self, inputs, neuron_names, output_neuron, connections):
    self.inputs = list(inputs)
    self.connections = connections
    self.neurons = {name:Neuron(name, [(from_neuron, weight) for (from_neuron, to_neuron, weight) in connections if to_neuron == name]) for name in neuron_names}
    # self.output_neurons = [neuron for neuron in output_neurons if neuron in self.neurons else raise Exception(str(neuron) + ' is an output, but not in neuron list!')]
    if output_neuron not in self.neurons:
      raise Exception(str(neuron) + ' is selected as output neuron, but is not in neuron list!')
    self.output_neuron = output_neuron
    self.top_sorted_neurons = self.top_sort(neuron_names)
  
  def graph(self):
    G = nx.DiGraph()
    G.add_weighted_edges_from([(input, neuron_name, neuron.weight_dict[input]) for (neuron_name, neuron) in self.neurons.items() for input in neuron.inputs])
    display_stn(G)
#     G = nx.DiGraph()
#     G.add_weighted_edges_from([(input, neuron_name, neuron.weight_dict[input]) for (neuron_name, neuron) in self.neurons.items() for input in neuron.inputs])
#     pos=nx.spring_layout(G)
#     nx.draw(G,pos,node_size=1200)
#     labels = nx.get_edge_attributes(G,'weight')
#     nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, font_size=16)
#     nx.draw_networkx_labels(G,pos,font_size=16)
#     plt.show()

    
  # returns a topologically sorted list of neuron names in the neural net
  def top_sort(self, neuron_names):
    unsorted_neuron_names = set(neuron_names)
    sorted_neuron_names = []
    valid_inputs = set(self.inputs)
    num_unsorted_neurons = len(unsorted_neuron_names)
    while num_unsorted_neurons > 0:
      for neuron in list(unsorted_neuron_names):
        if all([input in valid_inputs for input in self.neurons[neuron].inputs]):
          sorted_neuron_names.append(neuron)
          valid_inputs.add(neuron)
          unsorted_neuron_names.remove(neuron)
      if len(unsorted_neuron_names) == num_unsorted_neurons:
        raise Exception(str(num_unsorted_neurons) + ' neurons are unreachable or have unreachable inputs!')
      num_unsorted_neurons = len(unsorted_neuron_names)
    return sorted_neuron_names
    
  # returns a list of neurons with inputs connected to the output of the given neuron
  def get_output_connections(self, neuron):
    return [to_neuron for (from_neuron, to_neuron, weight) in self.connections if from_neuron == neuron]

  # perform forward propogation on the neural net, returning the overall output and a dictionary of each neuron's output


def plot_net(net, title, threshold_function=step):
    inputs = [input for input in net.inputs if str(input) == input]
    if len(inputs) == 1:
        inputs.append('?')
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    divisions = 100
    d = {inputs[0]:1, inputs[1]:1}
    m = np.zeros((divisions+1, divisions+1))
    for x in range(divisions + 1):
        for y in range(divisions + 1):
            d[inputs[0]] = x * float(x_max) / divisions + x_min
            d[inputs[1]] = y * float(y_max) / divisions + y_min
            m[y,x] = myForwardProp(net,d.copy(), threshold_function)[0]
    # plt.imshow(map, cmap='Greys',  interpolation='nearest')
    # plt.imshow(map, cmap='Blues',  interpolation='nearest')
    plt.imshow(m, cmap='Blues',  interpolation='bicubic')
    plt.gca().invert_yaxis()
    import matplotlib.ticker as tkr
    format = tkr.FuncFormatter(lambda x,y: '{}'.format(x / float(divisions)))
    plt.gca().xaxis.set_major_formatter(format)
    plt.gca().yaxis.set_major_formatter(format)
    plt.xlabel(inputs[0], fontsize=28)
    plt.ylabel(inputs[1], fontsize=28)
    plt.title(title, fontsize=38)
    plt.show()

# perform forward propogation on the neural net, returning the overall output and a dictionary of each neuron's output
def myForwardProp(neuralNet, input_value_dict, threshold_function):
    # neuralNet is the Neural Net Class Instance
    # input_value_dict is a copy of the input dictionary, mapping input variables to values
    #raise NotImplementedError
    
    for neuron in neuralNet.top_sorted_neurons:
      neuralNet.neurons[neuron].call(input_value_dict, threshold_function)
    return input_value_dict[neuralNet.output_neuron], input_value_dict
#
## triangle and rectangle
#inputs = ['x', 'y', -1]
#neurons = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10']
#output_neuron = 'n10'
#connections = [('x', 'n1', -1),
#               ('y', 'n1', -1),
#               ( -1, 'n1', -1.5),
#               ('x', 'n2', 1),
#               ('y', 'n2', -1),
#               ( -1, 'n2', -.5),
#               ('x', 'n3', 0),
#               ('y', 'n3', 1),
#               ( -1, 'n3', .5),
#               ('n1', 'n4', 1),
#               ('n2', 'n4', 1),
#               ('n3', 'n4', 1),
#               ( -1, 'n4',  2.5),
#               ('x', 'n5', -1),
#               ( -1, 'n5', -.75),
#               ('x', 'n6', 1),
#               ( -1, 'n6', .25),
#               ('y', 'n7', -1),
#               ( -1, 'n7', -.375),
#               ('y', 'n8', 1),
#               ( -1, 'n8', .125),
#               ('n5', 'n9', 1),
#               ('n6', 'n9', 1),
#               ('n7', 'n9', 1),
#               ('n8', 'n9', 1),
#               ( -1, 'n9',  3.5),
#               ('n4', 'n10', 1),
#               ('n9', 'n10', 1),
#               ( -1, 'n10',  .5)]
#xor_net = NN(inputs, neurons, output_neuron, connections)
#plot_net(xor_net, 'X XOR Y', step)
##xor_net.graph()
#
## verify forward prop
#d = {'x':1, 'y':1}
#checks = [((.5,.75), 1), 
#          ((.5,.45), 0), 
#          ((.5,.25), 1), 
#          ((.5,.08), 0), 
#          ((.1,.75), 0), 
#          ((.9,.75), 0), 
#          ((.1,.25), 0), 
#          ((.9,.25), 0)]
#for ((x,y), out) in checks:
#    d['x'] = x
#    d['y'] = y
#    net_output = myForwardProp(xor_net,d, step)[0]
#    if net_output != out:
#      raise Exception('Failed on (' + str(x) + ', ' + str(y) + '): ' + str(out) + ', output was: ' + str(net_output))
#test_ok()
#
##define necessary aspects here
#
#
##confirm your answer by checking the resulting plot
#
##or_net = NN(inputs, neurons, output_neuron, connections)
##or_net.graph()
##plot_net(or_net, 'X OR Y', step)
#
#inputs = ['x', 'y', .5]
#neurons = ['n1']
#output_neuron = 'n1'
#connections = [('x', 'n1',  1),
#               ('y', 'n1',  1),
#               (.5 , 'n1', -1)]
#or_net = NN(inputs, neurons, output_neuron, connections)
#plot_net(or_net, 'X OR Y', step)
#
##define necessary aspects here
#
#
##confirm your answer by checking the resulting plot
#
##notX_net = NN(inputs, neurons, output_neuron, connections)
##notX_net.graph()
##plot_net(notX_net, 'NOT X', step)
#
#inputs = ['x', .5]
#neurons = ['n1']
#output_neuron = 'n1'
#connections = [('x', 'n1', -1),
#               (.5 , 'n1',  1)]
#not_net = NN(inputs, neurons, output_neuron, connections)
#plot_net(not_net, 'NOT X', step)
#
#
##define necessary aspects here
#
#
##confirm your answer by checking the resulting plot
#
##xor_net = NN(inputs, neurons, output_neuron, connections)
##xor_net.graph()
##plot_net(xor_net, 'X XOR Y', step)
#
#inputs = ['x', 'y', 1.5, .5]
#neurons = ['n1', 'n2', 'n3']
#output_neuron = 'n3'
#connections = [('x', 'n1', -1),
#               ('y', 'n1', -1),
#               (1.5, 'n1',  1),
#               ('x', 'n2',  1),
#               ('y', 'n2',  1),
#               (.5 , 'n2', -1),
#               ('n1', 'n3',  1),
#               ('n2', 'n3',  1),
#               (1.5 , 'n3', -1)]
#xor_net = NN(inputs, neurons, output_neuron, connections)
##xor_net.graph()
#plot_net(xor_net, 'X XOR Y', step)

#For the following functions:

# neuralNet is a Neural Net Class Instance
# input_value_dict is an input dictionary, mapping input variables to values
# output_dict needs to be a mapping every neuron in the network to its output

# Helper Function for backward_prop- returns a dictionary of deltas, one for each neuron in the network
def compute_deltas(neuralNet, input_value_dict, output_dict, desired_out):
    #raise NotImplementedError
    out = output_dict[neuralNet.output_neuron]
    deltas = {neuralNet.output_neuron:out * (1 - out) * (desired_out - out)}
    for neuron in reversed(neuralNet.top_sorted_neurons[:-1]):
      deltas[neuron] = output_dict[neuron] * (1 - output_dict[neuron]) * \
    sum([neuralNet.neurons[out_neuron].get_weight(neuron) * deltas[out_neuron] for out_neuron in neuralNet.get_output_connections(neuron)])
    return deltas

# Helper Function for backward_prop- updates all of the weights in the network, doesn't return anything
def update_weights(neuralNet, input_value_dict, output_dict, desired_out, r):
    #raise NotImplementedError
    deltas = compute_deltas(neuralNet,input_value_dict, output_dict, desired_out)
    for neuron in neuralNet.neurons.values():
      for inp in neuron.neuron_inputs:
        neuron.update_weight(inp, neuron.get_weight(inp) + deltas[neuron.name] * r * output_dict[inp])
    
    
# Perform backward propogation on the neural net, and changes the weights in the network - it doesn't return anything
def myBackwardProp(neuralNet, input_value_dict, desired_out, threshold_function=sigmoid, r=1, max_error=-.0001):
    #raise NotImplementedError
    out, output_dict = myForwardProp(neuralNet,input_value_dict, threshold_function)
    while (accuracy(desired_out, out) < max_error):
      update_weights(neuralNet,input_value_dict, output_dict, desired_out, r)
      out, output_dict = myForwardProp(neuralNet,input_value_dict, threshold_function)

# triangle and rectangle
inputs = ['x', 'y', -1]
neurons = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10']
output_neuron = 'n10'
connections = [('x', 'n1', -1),
               ('y', 'n1', -1),
               ( -1, 'n1', -1.5),
               ('x', 'n2', 1),
               ('y', 'n2', -1),
               ( -1, 'n2', -.5),
               ('x', 'n3', 0),
               ('y', 'n3', 1),
               ( -1, 'n3', .5),
               ('n1', 'n4', 1),
               ('n2', 'n4', 1),
               ('n3', 'n4', 1),
               ( -1, 'n4',  2.5),
               ('x', 'n5', -1),
               ( -1, 'n5', -.75),
               ('x', 'n6', 1),
               ( -1, 'n6', .25),
               ('y', 'n7', -1),
               ( -1, 'n7', -.375),
               ('y', 'n8', 1),
               ( -1, 'n8', .125),
               ('n5', 'n9', 1),
               ('n6', 'n9', 1),
               ('n7', 'n9', 1),
               ('n8', 'n9', 1),
               ( -1, 'n9',  3.5),
               ('n4', 'n10', 1),
               ('n9', 'n10', 1),
               ( -1, 'n10',  .5)]
xor_net = NN(inputs, neurons, output_neuron, connections)
plot_net(xor_net, 'X XOR Y', step)
# xor_net.graph()

# verify backward prop
input_value_dict = {'x':.5, 'y':.45}
desired_out = 1
checks = [('n10', 'n9', 1.0580876267998136),
          ('n10', 'n4', 1.750519436878936),
          ('n10', -1, 0.5),
          ('n8', 'y', 1.004405571242847),
          ('n8', -1, 0.125),
          ('n9', 'n8', 1.0448828563571808),
          ('n9', -1, 3.5),
          ('n9', 'n5', 1.0417815987312213),
          ('n9', 'n6', 1.0417815987312213),
          ('n9', 'n7', 1.024362381541987),
          ('n1', 'y', -0.9842872685460131),
          ('n1', 'x', -0.9825414094955703),
          ('n1', -1, -1.5),
          ('n2', 'y', -0.9842872685460131),
          ('n2', 'x', 1.0174585905044304),
          ('n2', -1, -0.5),
          ('n3', 'y', 1.0418161693022556),
          ('n3', 'x', 0.04646241033584111),
          ('n3', -1, 0.5),
          ('n4', 'n1', 1.3130856737676413),
          ('n4', 'n2', 1.3130856737676413),
          ('n4', 'n3', 1.162829074164861),
          ('n4', -1, 2.5),
          ('n5', 'x', -0.9942918288280811),
          ('n5', -1, -0.75),
          ('n6', 'x', 1.005708171171915),
          ('n6', -1, 0.25),
          ('n7', 'y', -0.993643573887871),
          ('n7', -1, -0.375)]

myBackwardProp(xor_net,input_value_dict, desired_out)
for (neuron_name, inp, weight) in checks:
  weight_out = xor_net.neurons[neuron_name].weight_dict[inp]
  if abs(weight_out - weight) > .001:
    raise Exception('Failed on ' + str(neuron_name) + ' with input ' + str(inp) + ' and weight ' + str(weight) + ', weight was: ' + str(weight_out))
test_ok()
