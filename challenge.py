#!/usr/bin/env python3
"""
challenge.py
============
Finds the fastest path through a subway system.
This version assumes that all segments are distinct.
"""
import networkx as nx
import sys
from unidecode import unidecode

def read_system(fd):
    line_freqs = {}
    graph = nx.MultiGraph()

    for line in fd:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[1] == 'Frequency':
            line_freqs[parts[0]] = int(parts[2])
        else:
            graph.add_node(parts[1])
            graph.add_node(parts[2])
            graph.add_edge(parts[1], parts[2], line=parts[0], weight=int(parts[3]))

    return graph


dashes = str.maketrans('-', '_')

def ascii(french_name):
    return unidecode(french_name).translate(dashes).lower()


def write_system_dot(graph):
    lines = []
    lines.append("graph metro {")

    for n, (station, attrs) in enumerate(graph.nodes_iter(data=True)):
        lines.append("  {} [label=\"{}\"]".format(
            ascii(station), station
        ))

    for n, (stn1, stn2, attrs) in enumerate(graph.edges_iter(data=True)):
        lines.append("  {} -- {} [label=\"{}m\",len={},minlen={},color={}]".format(
            ascii(stn1), ascii(stn2),
            attrs['weight'], attrs['weight'], attrs['weight'], attrs['line'].lower()
        ))

    lines.append("}")
    return "\n".join(lines)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as fd:
        graph = read_system(fd)
    print(write_system_dot(graph))
