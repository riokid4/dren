from pathlib import Path
from typing import Optional, Set
import yaml
import networkx as nx

DEFAULT_ONTOLOGY_PATH = Path(__file__).parent / "ais_ontology.yml"


class AISOntologyGraph(nx.DiGraph):
    """Ontology graph for AI Safety concepts.
    Edges point from specific_node -> parent_node (type='isa' or 'partof')."""

    def __init__(self, yml_path: Optional[str | Path] = None):
        super().__init__()
        path = Path(yml_path) if yml_path else DEFAULT_ONTOLOGY_PATH
        self._load_yml(path)

    def _load_yml(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for branch, info in data.get("branches", {}).items():
            nodes = info.get("nodes", {})
            root = info.get("root", branch)
            if root not in self:
                self.add_node(root, branch=branch, synonyms=[])

            for node, props in nodes.items():
                synonyms = props.get("synonyms", [])
                if node not in self:
                    self.add_node(node, branch=branch, synonyms=synonyms)
                else:
                    self.nodes[node]["synonyms"] = synonyms
                    self.nodes[node]["branch"] = branch

                parent = props.get("parent", root)
                if parent not in self:
                    self.add_node(parent, branch=branch, synonyms=[])
                # Edge points from child (specific) -> parent (general)
                self.add_edge(node, parent, type="isa")

                for part_parent in props.get("partof", []):
                    if part_parent not in self:
                        self.add_node(part_parent, branch=branch, synonyms=[])
                    self.add_edge(node, part_parent, type="partof")

    def get_parents(self, node_id: str) -> Set[str]:
        """Returns immediate parent nodes (1 hop up via outgoing edges)."""
        if node_id not in self:
            return set()
        return set(self.successors(node_id))

    def get_children(self, node_id: str) -> Set[str]:
        """Returns immediate child nodes (1 hop down via incoming edges)."""
        if node_id not in self:
            return set()
        return set(self.predecessors(node_id))

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Returns all transitive parent/ancestor nodes (all hops up to root)."""
        if node_id not in self:
            return set()
        return nx.descendants(self, node_id)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Returns True if `ancestor` is reachable from `descendant` along child->parent edges."""
        if descendant not in self or ancestor not in self:
            return False
        if ancestor == descendant:
            return False
        return nx.has_path(self, descendant, ancestor)

    def refinement_of(self, specific_id: str, general_id: str) -> bool:
        """Returns True if `specific_id` is equal to or a descendant of `general_id`."""
        return specific_id == general_id or self.is_ancestor(general_id, specific_id)
