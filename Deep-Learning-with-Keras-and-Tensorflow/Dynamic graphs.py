pip install networkx
pip install matplotlib

import networkx as nx
import matplotlib.pyplot as plt
# Create a dynamic graph
G = nx.Graph()
# Add nodes and edges to the graph
G.add_node("Input")
G.add_node("Condition Check")
G.add_node("Path 1 Layer 1")
G.add_node("Path 1 Layer 2")
G.add_node("Path 2 Layer 1")
G.add_node("Path 2 Layer 2")
G.add_node("Output")
# Add edges for dynamic flow
G.add_edge_from([("Input", "Condition Check"),
                ("Condition Check", "Path 1 Layer 1"),
                ("Condition Check", "Path 2 Layer 1"),
                ("Path 1 Layer 1", "Path 1 Layer 2"),
                ("Path 2 Layer 1", "Path 2 Layer 2"),
                ("Path 1 Layer 2", "Output"),
                ("Path 2 Layer 2", "Output")])

# Position nodes using a shell layout
pos = nx.shell_layout(G)
# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Dynamic Graph Visualization")
plt.show()