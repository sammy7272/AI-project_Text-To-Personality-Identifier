"""
Module for graph-based career recommendations
"""
import networkx as nx

class CareerGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the career graph with nodes and edges"""
        # TODO: Add more detailed career paths and relationships
        careers = ['Software Engineer', 'Data Scientist', 'Product Manager',
                  'UX Designer', 'Business Analyst', 'Project Manager']
        
        # Add nodes
        for career in careers:
            self.graph.add_node(career)
        
        # Add edges (relationships between careers)
        self.graph.add_edge('Software Engineer', 'Data Scientist')
        self.graph.add_edge('Software Engineer', 'Product Manager')
        self.graph.add_edge('Data Scientist', 'Business Analyst')
        self.graph.add_edge('Product Manager', 'Project Manager')
        self.graph.add_edge('UX Designer', 'Product Manager')
    
    def get_recommendations(self, current_role, depth=2):
        """Get career recommendations based on current role"""
        if current_role not in self.graph:
            return []
        
        recommendations = []
        for node in nx.bfs_tree(self.graph, current_role, depth_limit=depth):
            if node != current_role:
                recommendations.append(node)
        
        return recommendations 