#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for visualizating STN's and distance graphs
"""

import os
import tempfile
import networkx as nx
import subprocess
from IPython.display import SVG, display
from utils import *

def display_weighted_graph(g):
    display(SVG(graph_to_svg(g, writer_fn=write_weighted_graph_to_dot)))

def display_stn(g):
    display(SVG(graph_to_svg(g, writer_fn=write_stn_to_dot)))


def write_stn_to_dot(g, f):
    #dpi=90
    f.write("digraph G {\n  rankdir=LR;\ndpi=60\n")
    # Nodes
    for v in g.nodes():
        f.write("  \"{}\" [shape=circle, width=0.6, fixedsize=true];\n".format(v))
    # Simple temporal constraints
    for (u, v) in g.edges():
        # stc = g[u][v]['stc']
        stc = g[u][v]['weight']
        f.write("  \"{}\" -> \"{}\" [label=\"{}\"];".format(u, v, str(round(stc,3))))        
#        f.write(str(round(stc,3)))
#        stc = g[u][v]['stc']
#        stc = (stc, stc)
#        f.write("  \"{}\" -> \"{}\" [label=\"{}\"];".format(u, v, format_stc(stc)))
    f.write("}")


def write_weighted_graph_to_dot(g, f):
    #dpi=90
    f.write("digraph G {\n  rankdir=LR;\ndpi=60\n")
    # Nodes
    for v in g.nodes():
        f.write("  \"{}\" [shape=circle, width=0.6, fixedsize=true];\n".format(v))
    # Simple temporal constraints
    for (u, v) in g.edges():
        w = g[u][v]['weight']
        f.write("  \"{}\" -> \"{}\" [label=\"{}\"];".format(u, v, format_num(w)))
    f.write("}")

def format_stc(stc):
    return "[{}, {}]".format(format_num(stc[0]), format_num(stc[1]))

def graph_to_svg(g, writer_fn=write_weighted_graph_to_dot):
    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
        name = f.name
        writer_fn(g, f)
    # Call dot on the name
    svg_file = "{}.svg".format(name)
    result = subprocess.call(["dot", "-Tsvg", name, "-o", svg_file])
    with open(svg_file,"r") as f:
        svg = f.read()

    os.remove(name)
    os.remove(svg_file)
    return svg
