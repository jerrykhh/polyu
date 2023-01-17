import argparse
from io import TextIOWrapper
from pathlib import Path
from uuid import uuid4
import re
from html.parser import HTMLParser
import pygraphviz

class Node:
    
    def __init__(self, id, value) -> None:
        self.id = id
        self.value = value

class Edge:
    
    def __init__(self, parent, child) -> None:
        self.parent= parent
        self.child = child

class DOMTree:
    
    def __init__(self, value: str, nodes: list=None) -> None:
        self.value: str = value
        self.nodes = nodes
        
    
    def add_node(self, node):
        if self.nodes is None:
            self.nodes = []
        self.nodes.append(node)
    
    def print(self, depth:int = 0):
        if depth > 0:
            print('  '*depth+'->', self.value)
        else:
            print(self.value)
        if self.nodes is not None:
            for node in self.nodes:
                node.print(depth+1)
    
    def get_all_nodes(self) -> list[Node]:
        if self.nodes is not None:
            node_tag = []
            for node in self.nodes:
                node_tag += node.get_all_nodes()
            return [Node(self.id, self.value)] + node_tag
        return [Node(self.id, self.value)]
    
    
    def get_all_edges(self, parent=None) -> list[Edge]:
        
        if self.nodes is not None:
            node_tag = []
            for node in self.nodes:
                node_tag += node.get_all_edges(self.id)
            return [Edge(parent=parent, child=self.id)] + node_tag
        return [Edge(parent=parent, child=self.id)]
        

class Parser(HTMLParser):
    
    def __init__(self, *, convert_charrefs: bool = ...) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        self.tmp_start_trees: list[DOMTree] = []
        self.dom_tree = None
        
    def handle_starttag(self, tag: str, attrs) -> None:
        # print("start", tag)
        tree = DOMTree(tag)
        if len(self.tmp_start_trees) > 0:
            self.tmp_start_trees[-1].add_node(tree)
        
        self.tmp_start_trees.append(tree)      
    
    def handle_endtag(self, tag: str) -> None:
        # print("end", tag)
        index = len(self.tmp_start_trees)-1
        while(index > 0):
            
            tree = self.tmp_start_trees.pop()
            if tree.value == tag:
                break
        
        if index == 0:
            self.dom_tree = self.tmp_start_trees[0]
            
            
def init():
    
    parser = argparse.ArgumentParser(description="Draw the DOM tree for the <body> element of the HTML")
    parser.add_argument("-html", type=str, required=True, help="html file path")
    parser.add_argument("-output", type=str, default=None, help="Path of file (optional)")
    args = parser.parse_args()
    
    if args.output is None:
        default_output_path = f"{Path(__file__).parent.absolute()}/"
        Path(default_output_path).mkdir(exist_ok=True)
        args.output = f"{default_output_path}{uuid4().hex[:8]}.png"
    
    html = html_preprocessing(open(args.html, mode='r'))
    
    convert2DOMTree(html, args.output)    

def html_preprocessing(html_fp: TextIOWrapper) -> str:
    
    html = html_fp.read()
    html = html.replace("\n", "").replace("\t", "")
    html = re.search("<body.*>.*</body>", html)
    if html is None:
        print("<body> tag is not found!")
        exit(1)
    return html.group(0)
    
    

def convert2DOMTree(html: str, DOMTree_img_path:str):
    
    htmlParse = Parser()
    htmlParse.feed(html)
    G = pygraphviz.AGraph(directed=True, strict=False, ranksep=0.2, splines="spline", concentrate=True)
    dom_tree = htmlParse.dom_tree
    dom_tree.print()
    
    nodes = dom_tree.get_all_nodes()
    for node in nodes:
        G.add_node(node.id, label=f"{node.value}", color="black", style="solid", penwidth=1, fontname="Times-Roman", fontsize=14, fontcolor="black",
           arrowsize=1, arrowhead="normal", arrowtail="normal", dir="forward", shape="polygon")
    
    edges = dom_tree.get_all_edges()
    for edge in edges:
        if edge.parent is not None:
            G.add_edge([edge.parent, edge.child], color="#7F01FF", arrowsize=0.8)
    G.layout()
    G.draw(DOMTree_img_path, prog="dot")
    print(f"{DOMTree_img_path} saved")

    
if __name__ == "__main__":
    init()