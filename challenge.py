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

Line = namedtuple('Line', ['name', 'frequency', 'color', 'stations', 'segments'])
Segment = namedtuple('Segment', ['lines', 'left', 'right', 'time'])


def read_system(fd):
    """
    Reads a system description file in a way that it can be converted to
    many different kinds of graphs.

    It returns a tuple `line_freqs` (a mapping of line names to frequencies,
    in minutes), `station_lines` (a mapping of station names to sets of line
    names), and `line_segments` (a mapping of line names to lists of
    (station, station) pairs).
    """
    lines = {}

    for line in fd:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[1] == 'Frequency':
            # Line declaration
            line, _freq_, freq = parts
            lines[line] = Line(line, int(freq), line.lower(), [], [])
        else:
            # Line initialization
            line, stn1, stn2, time = parts
            stations, segments = lines[line].stations, lines[line].segments

            if not stations:
                stations.append(stn1)
            elif stations[-1] != stn1:
                raise ValueError("Stations must be listed sequentially")

            stations.append(stn2)
            segments.append(Segment(line, stn1, stn2, int(time)))

    return lines


###
### GraphViz Helpers
###

dashes = str.maketrans('-', '_')

def ascii(french_name):
    return unidecode(french_name).translate(dashes).lower()


###
### Digraph Model
###

#: A node is a (line, station, direction) pair -- it represents
#: a platform at a station.
#: This allows both track segments and transfers to be represented as edges.
Node = namedtuple('Node', ['line', 'station', 'direction'])


#: Represents riding a train across a section of track.
RIDE = "ride"

#: Represents switching direction at the end of the line.
TURNAROUND = "turnaround"

#: Represents transferring from one train to another at a transfer station.
TRANSFER = "transfer"


def build_digraph(lines):
    graph = nx.DiGraph()

    transfer_from = {}
    transfer_to = {}

    # Construct the lines!
    # We do each line separately.
    for line in lines.values():
        # Find the directions!
        head_station, tail_station = line.stations[0], line.stations[-1]

        # Build the first station!
        prev_node_head = Node(line.name, head_station, head_station)
        prev_node_tail = Node(line.name, head_station, tail_station)
        prev_station = head_station

        # (Also add it to the transfer_from list.)
        transfer_from.setdefault(head_station, set()).add((line.name, head_station))
        transfer_to.setdefault(head_station, set()).add((line.name, tail_station))

        graph.add_node(prev_node_head)
        graph.add_node(prev_node_tail)

        # Iterate through and build the subsequent stations and lines!
        for n in range(len(line.stations) - 1):
            new_station = line.stations[n + 1]
            segment = line.segments[n]

            # Build the station nodes!
            new_node_head = Node(line.name, new_station, head_station)
            new_node_tail = Node(line.name, new_station, tail_station)

            # Connect prev_station and next_station!
            graph.add_edge(prev_node_tail, new_node_tail,
                           action=RIDE, weight=segment.time, target=segment)
            graph.add_edge(new_node_head, prev_node_head,
                           action=RIDE, weight=segment.time, target=segment)

            # Add it to the transfer lists!
            transfer_from.setdefault(new_station, set()).add((line.name, tail_station))
            transfer_to.setdefault(new_station, set()).add((line.name, head_station))

            # Add it to the transfer lists, in the other direction!
            if new_station != tail_station:
                transfer_from[new_station].add((line.name, head_station))
                transfer_to[new_station].add((line.name, tail_station))

            prev_node_head, prev_node_tail = new_node_head, new_node_tail

        # Build the turnarounds!
        # One connects the tail station from tail-direction to head.
        # One connects the head station from head-direction to tail.
        graph.add_edge(Node(line.name, tail_station, tail_station),
                       Node(line.name, tail_station, head_station),
                       action=TURNAROUND, weight=line.frequency)
        graph.add_edge(Node(line.name, head_station, head_station),
                       Node(line.name, head_station, tail_station),
                       action=TURNAROUND, weight=line.frequency)

    # Now, build the transfers!
    # (We can iterate over either transfer_from or transfer_to,
    # since entries are always created in pairs.)
    for station in transfer_from.keys():
        transfers = itertools.product(transfer_from[station], transfer_to[station])
        for (from_line, from_dir), (to_line, to_dir) in transfers:
            if from_line == to_line:
                continue

            graph.add_edge(Node(from_line, station, from_dir),
                           Node(to_line, station, to_dir),
                           action=TRANSFER, weight=lines[to_line].frequency)

    return graph

    ####
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


def node_identifier(node):
    return ascii(node.line) + "_" + ascii(node.station) + "_" + ascii(node.direction)


def write_digraph_dot(graph):
    lines = []
    lines.append("digraph metro {")

    for node in graph.nodes_iter():
        lines.append("  {} [label=\"{}\", color={}]".format(
            node_identifier(node),
            node.station + "\\n" + node.line + "\\nto " + node.direction,
            node.line.lower()
        ))

    for node1, node2, data in graph.edges_iter(data=True):
        time = data['weight']

        if data['action'] == TRANSFER:
            color = 'black'
            label = "Wait {}m".format(time)
            style = "solid"
        elif data['action'] == TURNAROUND:
            color = node1.line.lower()
            label = "Wait {}m".format(time)
            style = "dashed"
        else:
            color = node1.line.lower()
            label = "{}m".format(time)
            style = "solid"

        lines.append("  {} -> {} [".format(node_identifier(node1),
                                           node_identifier(node2)))
        #lines.append("    len={}, minlen={},".format(time, time))
        lines.append("    label=\"{}\", color={}, style={}".format(label, color, style))
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

        # Precalculate edge ordering
        # Put RIDE/TURNAROUND edges before TRANSFER edges, then sort by weight
        self.outbound_edges = {}
        ranks = {RIDE: 0, TURNAROUND: 0, TRANSFER: 1}
        edge_key = lambda e: (ranks[e[2]['action']], e[2]['weight'], e[0], e[1])

        for node in graph:
            edges = list(graph.out_edges((node,), data=True))
            edges.sort(key=edge_key)
            self.outbound_edges[node] = edges

        # Results
        self.minimum_time = None
        self.minimum_paths = []

    def run(self):
        if self.has_run:
            raise RuntimeError("PathFinders are only usable once")

        # Create some starting positions
        empty_path = Path((), frozenset(), 0, self.targets)
        for node in sorted(self.graph):
            self.queue.appendleft((empty_path, node))

        # Recursively investigate
        count = 0
        while len(self.queue) > 0 and count < 10000000000000:
            path, at_node = self.queue.popleft()
            self.investigate(path, at_node)
            count = count + 1

    def investigate(self, path, at_node):
        options = self.outbound_edges[at_node]

        for edge in options:
            # Don't try a path if it's going to be longer than
            # our current minimum.
            if (
                self.minimum_time is not None and
                path.time + edge[2]['weight'] > self.minimum_time
            ):
                continue

            # Don't repeat edges.
            if edge[:2] in path.used_edges:
                continue

            # Don't transfer if we just transferred.
            if path.edges:
                prev_edge = path.edges[-1]
                if prev_edge[2]['action'] == TRANSFER and edge[2]['action'] == TRANSFER:
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
                self.queue.appendleft((new_path, new_node))

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
        lines = read_system(fd)

    digraph = build_digraph(lines)

    command = sys.argv[2] if len(sys.argv) > 2 else None

    if command == '--graph':
        print(write_digraph_dot(digraph))
    elif command is None:
        finder = PathFinder(digraph)
        finder.run()

        for path in finder.minimum_paths:
            print("In {} minutes:".format(path.time))
            for edge in path.edges:
                time = edge[2]['weight']
                print("[{:>3}m] ".format(time), end='')

                if edge[2]['action'] == RIDE:
                    print("Ride from      {:<18} to {:<18} on {:<10} (direction {})".format(
                        edge[0].station, edge[1].station, edge[0].line, edge[0].direction
                    ))
                elif edge[2]['action'] == TURNAROUND:
                    print("Turn around at {:<18} {:<21} on {:<10} (direction {})".format(
                        edge[1].station, '', edge[1].line, edge[1].direction
                    ))
                else:
                    print("Transfer at    {:<18} {:<21} to {:<10} (direction {})".format(
                        edge[0].station, '', edge[1].line, edge[1].direction
                    ))
            print()

