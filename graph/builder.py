import networkx as nx
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'raw', 'support_tickets.json')

class SupportGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.load_graph()

    def load_graph(self):
        if not os.path.exists(RAW_PATH):
            print("No raw data found to build graph.")
            return

        with open(RAW_PATH, 'r') as f:
            tickets = json.load(f)

        for t in tickets:
            product = f"Product:{t['product']}"
            category = f"Category:{t['category']}"
            issue = f"Issue:{t['id']}"
            
            # Add nodes
            self.G.add_node(product, type='product', name=t['product'])
            self.G.add_node(category, type='category', name=t['category'])
            self.G.add_node(issue, type='issue', desc=t['issue'], resolution=t['resolution'])
            
            # Add edges
            self.G.add_edge(product, category, relation='has_category')
            self.G.add_edge(category, issue, relation='has_issue')
            self.G.add_edge(product, issue, relation='related_to')

    def get_related_issues(self, product_name):
        """Returns a summary of common issues for a product."""
        prod_node = f"Product:{product_name}"
        if not self.G.has_node(prod_node):
            return "No information found in the knowledge graph for this product."
            
        issues = []
        for neighbor in self.G.neighbors(prod_node):
            if self.G.nodes[neighbor].get('type') == 'issue':
                issues.append(self.G.nodes[neighbor]['desc'])
                if len(issues) >= 5:
                    break
                
        if not issues:
            return "No known issues linked to this product."
            
        return f"Common issues for {product_name}: " + "; ".join(issues)

# Singleton instance for the application
support_graph = SupportGraph()

if __name__ == "__main__":
    print(f"Graph nodes: {support_graph.G.number_of_nodes()}")
    print(f"Graph edges: {support_graph.G.number_of_edges()}")
    print(support_graph.get_related_issues("CloudStore Pro"))
