from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Text, Tuple
from enum import IntEnum

from src.stance import Stance


class GraphNodeCategory(IntEnum):
    Z = 1
    NZ = 2
    FAQ = 3
    OTHER = 4

    @staticmethod
    def from_string(category_string: Text) -> GraphNodeCategory:
        string_to_category = {
            "z": GraphNodeCategory.Z,
            "nz": GraphNodeCategory.NZ,
            "faq": GraphNodeCategory.FAQ,
        }

        category_string = category_string.lower().strip()
        category = string_to_category.get(category_string, GraphNodeCategory.OTHER)
        return category


@dataclass
class GraphNode:
    label: Text
    full_text: Text = field(repr=False)
    parents: Set[Text] = field(default_factory=set)
    label_name: Text = field(default="")
    paraphrase: Text = field(default="", repr=False)
    summary: Text = field(default="")
    groups: Set[Text] = field(default_factory=set)
    rating: int = field(default=-1, repr=False)
    reaction_to_argument: int = field(default=-1, repr=False)
    category: GraphNodeCategory = field(default=GraphNodeCategory.OTHER, repr=False)
    stance: Stance = field(default=Stance.OTHER, repr=False)
    samples: Set[Text] = field(default_factory=set, repr=False)
    similars: Set[Text] = field(default_factory=set)

    child_nodes: Set[GraphNode] = field(init=False, default_factory=set, repr=False)
    parent_nodes: Set[GraphNode] = field(init=False, default_factory=set, repr=False)
    similar_nodes: Set[GraphNode] = field(init=False, default_factory=set, repr=False)
    group_nodes: Set[GraphNode] = field(init=False, default_factory=set, repr=False)
    group_member_nodes: Set[GraphNode] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._cleanup_attributes()
        self._match_constraints()

    def __repr__(self) -> str:
        label = getattr(self, "label", "undefined")
        return f"{self.__class__.__name__}(label={label!r})"

    def _cleanup_attributes(self) -> None:
        if not hasattr(self, 'label'):
            raise ValueError("A Node object must have a 'label' attribute.")
        self.label = self.label.strip()

        self.parents = set(parent.strip() for parent in self.parents)
        self.groups = set(group.strip() for group in self.groups)

        self.label_name = self.label_name.strip()
        if self.label_name == "":
            self.label_name = self.label

    def _match_constraints(self) -> None:
        if self.label in self.groups:
            print(
                f"WARNING: group labels must not contain node itself -> label: {self.label}, groups: {self.groups}"
            )
            self.groups.discard(self.label)
        if self.label in self.parents:
            print(
                f"WARNING: parent labels must not contain node itself -> "
                f"label: {self.label}, parent_label: {self.parents}"
            )
            self.parents.discard(self.label)

    @property
    def children(self) -> Set[Text]:
        return set(node.label for node in self.child_nodes)

    @property
    def has_child_labels(self) -> bool:
        return len(self.children) > 0

    @property
    def has_child_nodes(self) -> bool:
        return len(self.child_nodes) > 0

    @property
    def has_parent_labels(self) -> bool:
        return len(self.parents) > 0

    @property
    def has_parent_nodes(self) -> bool:
        return len(self.parent_nodes) > 0

    @property
    def has_group_labels(self) -> bool:
        return len(self.groups) > 0

    @property
    def has_group_nodes(self) -> bool:
        return len(self.group_nodes) > 0

    @property
    def group_members(self) -> Set[Text]:
        return set(node.label for node in self.group_member_nodes)

    @property
    def has_group_member_labels(self) -> bool:
        return len(self.group_members) > 0

    @property
    def has_group_member_nodes(self) -> bool:
        return len(self.group_member_nodes) > 0

    @property
    def super_group_nodes(self) -> Set[GraphNode]:
        """Returns a set of all group nodes above the node."""
        super_group_nodes = set()
        for group_node in self.group_nodes:
            if group_node in super_group_nodes:
                continue
            super_group_nodes.add(group_node)
            super_group_nodes.update(group_node.super_group_nodes)
        return super_group_nodes

    @property
    def sub_group_member_nodes(self) -> Set[GraphNode]:
        """Returns a set of all group member nodes under the node."""
        sub_group_member_nodes = set()
        for group_member_node in self.group_member_nodes:
            if group_member_node in sub_group_member_nodes:
                continue
            sub_group_member_nodes.add(group_member_node)
            sub_group_member_nodes.update(group_member_node.sub_group_member_nodes)
        return sub_group_member_nodes

    @property
    def ancestor_nodes(self) -> Set[GraphNode]:
        ancestor_nodes = set()
        for parent_node in self.parent_nodes:
            if parent_node in ancestor_nodes:
                continue
            ancestor_nodes.add(parent_node)
            ancestor_nodes.update(parent_node.ancestor_nodes)
        return ancestor_nodes

    @property
    def descendant_nodes(self) -> Set[GraphNode]:
        descendant_nodes = set()
        for child_node in self.child_nodes:
            if child_node in descendant_nodes:
                continue
            descendant_nodes.add(child_node)
            descendant_nodes.update(child_node.descendant_nodes)
        return descendant_nodes

    def as_tuple(self) -> Tuple:
        return self.label, self.summary, self.paraphrase, self.full_text, self.rating, next(iter(self.groups))

    def as_labelfile_dict(self) -> Text:
        return self.label

    def __hash__(self) -> int:
        if hasattr(self, 'label'):
            return hash(self.label)
        else:
            return hash(None)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, GraphNode):
            return hash(self) == hash(__o)
        return False


if __name__ == "__main__":
    print("test")
