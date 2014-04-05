#!/usr/bin/env python3
"""
challenge.py
============
Finds the fastest path through a subway system.
This version assumes that all segments are distinct.
"""
import itertools
import networkx as nx
import pprint
import sys
from collections import namedtuple, deque
from unidecode import unidecode


###
### File Loading
###

def read_system(fd):
    """
    Reads a system description file in a way that it can be converted to
    many different kinds of graphs.

    It returns a tuple `line_freqs` (a mapping of line names to frequencies,
    in minutes), `station_lines` (a mapping of station names to sets of line
    names), and `line_segments` (a mapping of line names to lists of
    (station, station) pairs).
    """
    line_freqs = {}
    station_lines = {}
    line_segments = {}

    for line in fd:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[1] == 'Frequency':
            line, _freq_, freq = parts
            line_freqs[line] = int(freq)
            line_segments.setdefault(line, set())
        else:
            line, stn1, stn2, time = parts
            line_segments[line].add((stn1, stn2, int(time)))
            station_lines.setdefault(stn1, set()).add(line)
            station_lines.setdefault(stn2, set()).add(line)

    return line_freqs, station_lines, line_segments


###
### GraphViz Helpers
###

dashes = str.maketrans('-', '_')

def ascii(french_name):
    return unidecode(french_name).translate(dashes).lower()


###
### Digraph Model
###

#: A node is a (line, station) pair -- it represents a platform at a station.
#: This allows both track segments and transfers to be represented as edges.
Node = namedtuple('Node', ['line', 'station'])


#: Represents riding a train across a section of track.
RIDE = "ride"

#: Represents transferring from one train to another at a transfer station.
TRANSFER = "transfer"


def build_digraph(line_freqs, station_lines, line_segments):
    graph = nx.DiGraph()

    for station, lines in station_lines.items():
        # Build the stations
        for line in lines:
            graph.add_node(Node(line, station))

        # Build the transfer edges
        for l1, l2 in itertools.combinations(lines, 2):
            # The weight (travel time) here is determined by
            # the frequency of the trains on the destination line
            # (we assume the worst)
            graph.add_edge(Node(l1, station), Node(l2, station),
                           weight=line_freqs[l2], action=TRANSFER)
            graph.add_edge(Node(l2, station), Node(l1, station),
                           weight=line_freqs[l1], action=TRANSFER)

    for line, segments in line_segments.items():
        # Build the metro lines
        for stn1, stn2, time in segments:
            # Determine the segment's "target class"
            # THE GOAL: ride one segment from each target class
            target = (line, stn1, stn2)

            # There needs to be a forward and reverse edge
            graph.add_edge(Node(line, stn1), Node(line, stn2),
                           weight=time, action=RIDE, target=target)
            graph.add_edge(Node(line, stn2), Node(line, stn1),
                           weight=time, action=RIDE, target=target)

    return graph


def write_digraph_dot(graph):
    lines = []
    lines.append("digraph metro {")

    for node in graph.nodes_iter():
        name = node.station + "\\n" + node.line
        ident = ascii(node.line) + "_" + ascii(node.station)
        lines.append("  {} [label=\"{}\"]".format(ident, name))

    for node1, node2, data in graph.edges_iter(data=True):
        time = data['weight']
        ident1 = ascii(node1.line) + "_" + ascii(node1.station)
        ident2 = ascii(node2.line) + "_" + ascii(node2.station)

        if data['action'] == TRANSFER:
            color = 'black'
            label = "Wait {}m".format(time)
        else:
            color = node1.line.lower()
            label = "{}m".format(time)

        lines.append("  {} -> {} [".format(ident1, ident2))
        lines.append("    len={}, minlen={},".format(time, time))
        lines.append("    label=\"{}\", color={}".format(label, color))
        lines.append("  ]")

    lines.append("}")

    return '\n'.join(lines)


###
### The Brute-Force Approach
###

Path = namedtuple('Path', ['edges', 'used_edges', 'time', 'targets_left'])


def path_advance(path, edge):
    if 'target' in edge[2]:
        targets_left = path.targets_left - {edge[2]['target']}
    else:
        targets_left = path.targets_left

    return Path(path.edges + (edge,), path.used_edges | {edge[:2]},
                path.time + edge[2]['weight'], targets_left)


class PathFinder:
    def __init__(self, graph):
        # State
        self.has_run = False
        self.queue = deque()

        # Analyze the graph
        self.graph = graph
        self.targets = frozenset(
            data['target'] for n1, n2, data in graph.edges(data=True) if 'target' in data
        )

        # Results
        self.minimum_time = None
        self.minimum_paths = []

    def run(self):
        if self.has_run:
            raise RuntimeError("PathFinders are only usable once")

        # Create some starting positions
        empty_path = Path((), frozenset(), 0, self.targets)
        for node in self.graph:
            self.queue.append((empty_path, node))

        # Recursively investigate
        while len(self.queue) > 0:
            path, at_node = self.queue.popleft()
            self.investigate(path, at_node)

    def investigate(self, path, at_node):
        options = list(self.graph.out_edges((at_node,), data=True))
        current_time = path.time

        for edge in options:
            # Don't try a path if it's going to be longer than
            # our current minimum.
            if (
                self.minimum_time is not None and
                current_time + edge[2]['weight'] > self.minimum_time
            ):
                continue

            # Don't repeat edges.
            if edge[:2] in path.used_edges:
                continue

            # Don't turn around unless it's our only option.
            if path.edges and len(options) > 1:
                prev_edge = path.edges[-1]
                if prev_edge[0] == edge[1] and prev_edge[1] == edge[0]:
                    continue

            # Try it!
            new_node = edge[1]
            new_path = path_advance(path, edge)
            if not new_path.targets_left:
                # We win! This path gets us all the way through the metro.
                # Add it to our success list!
                self.found(new_path)
            else:
                # Keep following the path!
                self.queue.append((new_path, new_node))

    def found(self, path):
        if self.minimum_time is None or path.time < self.minimum_time:
            print("Found a path that takes {} minutes!".format(path.time))
            self.minimum_time = path.time
            self.minimum_paths = [path]
        elif path.time == self.minimum_time:
            print("Found another path that takes {} minutes!".format(path.time))
            self.minimum_paths.append(path)
        else:
            pass


###
### And, the main code
###

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as fd:
        freqs, stns, segments = read_system(fd)

    digraph = build_digraph(freqs, stns, segments)

    command = sys.argv[2] if len(sys.argv) > 2 else None

    if command == '--graph':
        print(write_digraph_dot(digraph))
    elif command is None:
        finder = PathFinder(digraph)
        finder.run()

        for path in finder.minimum_paths:
            print("In {} minutes:".format(path.time))
            for edge in path.edges:
                if edge[2]['action'] == RIDE:
                    print("Ride {} from {} to {} [{}m]".format(
                        edge[0].line, edge[0].station, edge[1].station, edge[2]['weight']
                    ))
                else:
                    print("Transfer from {} to {} at {} [{}m]".format(
                        edge[0].line, edge[1].line, edge[0].station, edge[2]['weight']
                    ))
            print()

