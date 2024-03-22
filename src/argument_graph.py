from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Set, Text, Optional, Tuple
from cached_property import cached_property
import pandas as pd

from src.read_argument_graph import (
    load_samples_df_from_file,
    load_nodes_df_from_file,
    get_nodes_from_df,
)
from src.graph_node import GraphNode, GraphNodeCategory
from src.stance import Stance


@dataclass
class ArgumentGraph:
    introduction_nodes: List[GraphNode]
    transition_nodes: List[GraphNode]
    z_arguments_nodes: List[GraphNode]
    group_nodes: List[GraphNode]
    nz_arguments_nodes: List[GraphNode]
    faq_nodes: List[GraphNode]

    def __post_init__(self) -> None:
        self._label_to_node = {node.label: node for node in self.all_nodes}
        self._init_node_references()
        self._check_constraints()

    def _init_node_child_and_parent_references(self) -> None:
        for node in self.all_nodes:
            if not node.has_parent_labels:
                continue
            parent_nodes = [self.get_node_for_label(parent_label) for parent_label in node.parents]
            parent_nodes = [node for node in parent_nodes if node is not None]
            node.parent_nodes.update(parent_nodes)

            # add node as child for parent nodes
            for parent_node in parent_nodes:
                parent_node.child_nodes.add(node)

    def _init_node_similars_references(self) -> None:
        for node in self.all_nodes:
            if not node.similars:
                continue

            similar_nodes = [self.get_node_for_label(similar_label) for similar_label in node.similars]
            similar_nodes = [node for node in similar_nodes if node is not None]
            node.similar_nodes.update(similar_nodes)

            # add node as similar node for the other nodes
            for similar_node in similar_nodes:
                similar_node.similar_nodes.add(node)

    def _init_node_group_references(self) -> None:
        for node in self.all_nodes:
            if not node.has_group_labels:
                continue

            # search for group nodes and add them to the node
            group_nodes = [self.get_node_for_label(group_label) for group_label in node.groups]
            group_nodes = [node for node in group_nodes if node is not None]
            node.group_nodes.update(group_nodes)

            # for each found group node add the node to their group member lists
            for group_node in group_nodes:
                group_node.group_member_nodes.add(node)

    def _init_node_references(self) -> None:
        self._init_node_child_and_parent_references()
        self._init_node_group_references()
        self._init_node_similars_references()

    def _check_constraints(self) -> None:
        for node in self.all_nodes:
            # check that each parent node has node as child node
            for parent_node in node.parent_nodes:
                assert node in parent_node.child_nodes

            # check that each child node has node as parent node
            for child_node in node.child_nodes:
                assert node in child_node.parent_nodes

            # check that each group node has node as group member node
            for group_node in node.group_nodes:
                assert node in group_node.group_member_nodes

            # check that each group member node has node as group node
            for group_member_node in node.group_member_nodes:
                assert node in group_member_node.group_nodes

            # check that each similar node has node as similar node
            for similar_node in node.similar_nodes:
                assert node in similar_node.similar_nodes

    @cached_property
    def introduction_labels(self) -> Set[Text]:
        """Returns a set of all labels for introduction nodes."""
        return self._label_set_from_nodes(self.introduction_nodes)

    @cached_property
    def transition_labels(self) -> Set[Text]:
        """Returns a set of all labels for transition nodes."""
        return self._label_set_from_nodes(self.transition_nodes)

    @cached_property
    def z_arguments_labels(self) -> Set[Text]:
        """Returns a set of all labels for z argument nodes."""
        return self._label_set_from_nodes(self.z_arguments_nodes)

    @cached_property
    def group_labels(self) -> Set[Text]:
        """Returns a set of all labels for group nodes."""
        return self._label_set_from_nodes(self.group_nodes)

    @cached_property
    def nz_arguments_labels(self) -> Set[Text]:
        """Returns a set of all labels for nz argument nodes."""
        return self._label_set_from_nodes(self.nz_arguments_nodes)

    @cached_property
    def faq_labels(self) -> Set[Text]:
        """Returns a set of all labels for faq nodes."""
        return self._label_set_from_nodes(self.faq_nodes)

    @property
    def all_nodes(self) -> List[GraphNode]:
        """
        Concatenates all node lists in the collection and returns them as one big list.
        """
        return (
                self.introduction_nodes
                + self.transition_nodes
                + self.z_arguments_nodes
                + self.group_nodes
                + self.nz_arguments_nodes
                + self.faq_nodes
        )

    @cached_property
    def all_labels(self) -> Set[Text]:
        """Returns a set of all labels from all nodes."""
        return self._label_set_from_nodes(self.all_nodes)

    @property
    def arguments_nodes(self) -> List[GraphNode]:
        """
        Concatenates all zarg and nzarg node lists in the collection and returns them as one big list.
        """
        return self.z_arguments_nodes + self.nz_arguments_nodes

    @property
    def user_arguments_nodes(self) -> List[GraphNode]:
        return self.z_arguments_nodes + self.nz_arguments_first_level_nodes

    @cached_property
    def user_arguments_labels(self) -> Set[Text]:
        return self._label_set_from_nodes(self.user_arguments_nodes)

    @property
    def user_intent_nodes(self) -> List[GraphNode]:
        return self.user_arguments_nodes + self.faq_question_nodes

    @cached_property
    def user_intent_labels(self) -> Set[Text]:
        return self._label_set_from_nodes(self.user_intent_nodes)

    @cached_property
    def arguments_labels(self) -> Set[Text]:
        """Returns a set of all labels for arguments nodes."""
        return self._label_set_from_nodes(self.arguments_nodes)

    @property
    def faq_question_nodes(self) -> List[GraphNode]:
        """
        Returns only the faq nodes that represent the questions and not possible answers.
        """
        return [node for node in self.faq_nodes if not node.has_parent_labels]

    @cached_property
    def faq_question_labels(self) -> Set[Text]:
        """Returns a set of all labels for faq question nodes."""
        return self._label_set_from_nodes(self.faq_question_nodes)

    @property
    def faq_answer_nodes(self) -> List[GraphNode]:
        """
        Returns only the faq nodes that represent the answers and not the related questions.
        """
        return [node for node in self.faq_nodes if node.has_parent_labels]

    @property
    def faq_node_pairs(self) -> List[Tuple[GraphNode, GraphNode]]:
        """
        Returns question and answer pairs of the faq nodes.
        The tuple format is (question_node, answer_node).
        """
        q_a_tuples = []
        for answer_node in self.faq_answer_nodes:
            if len(answer_node.parent_nodes) < 1:
                print(f"WARNING: found FAQ answer node without question -> {answer_node}")
                continue

            if len(answer_node.parent_nodes) > 1:
                print(f"WARNING: found FAQ answer node for multiple questions -> {answer_node}")

            question_node = next(iter(answer_node.parent_nodes))
            q_a_tuples.append(tuple([question_node, answer_node]))

        return q_a_tuples

    @property
    def paraphrase_introduction_nodes(self) -> List[GraphNode]:
        def groups_contain_paraphrase(groups: Set[str]) -> bool:
            for group in groups:
                if "paraphrase" in group:
                    return True
            return False

        return [node for node in self.introduction_nodes if groups_contain_paraphrase(node.groups)]

    @property
    def z_nodes(self) -> List[GraphNode]:
        """Returns only nodes with the node category z."""
        return [node for node in self.all_nodes if node.category == GraphNodeCategory.Z]

    @cached_property
    def z_labels(self) -> Set[Text]:
        """Returns a set of all labels for z nodes."""
        return self._label_set_from_nodes(self.z_nodes)

    @property
    def nz_nodes(self) -> List[GraphNode]:
        """Returns only nodes with the node category nz."""
        return [node for node in self.all_nodes if node.category == GraphNodeCategory.NZ]

    @cached_property
    def nz_labels(self) -> Set[Text]:
        """Returns a set of all labels for nz nodes."""
        return self._label_set_from_nodes(self.nz_nodes)

    @property
    def nz_arguments_first_level_nodes(self) -> List[GraphNode]:
        return [node for node in self.nz_arguments_nodes if not node.has_parent_labels]

    @property
    def z_group_nodes(self) -> List[GraphNode]:
        """Returns only the z group nodes."""
        return [node for node in self.group_nodes if node.category == GraphNodeCategory.Z]

    @cached_property
    def z_group_labels(self) -> Set[Text]:
        """Returns a set of all labels for z group nodes."""
        return self._label_set_from_nodes(self.z_group_nodes)

    @property
    def nz_group_nodes(self) -> List[GraphNode]:
        """Returns only the nz group nodes."""
        return [node for node in self.group_nodes if node.category == GraphNodeCategory.NZ]

    @cached_property
    def nz_group_labels(self) -> Set[Text]:
        """Returns a set of all labels for nz group nodes."""
        return self._label_set_from_nodes(self.nz_group_nodes)

    @property
    def faq_group_nodes(self) -> List[GraphNode]:
        """Returns only the faq group nodes."""
        return [node for node in self.group_nodes if node.category == GraphNodeCategory.FAQ]

    @cached_property
    def faq_group_labels(self) -> Set[Text]:
        """Returns a set of all labels for faq group nodes."""
        return self._label_set_from_nodes(self.faq_group_nodes)

    @property
    def first_level_z_nodes(self) -> Set[GraphNode]:
        return set(filter(lambda node: not node.has_parent_labels, self.z_nodes))

    @property
    def primary_z_nodes(self) -> Set[GraphNode]:
        """Returns z nodes that have a rating of 0 or higher.
            Rating > 0 indicates that the node can be uttered by the bot. Rating = 0 indicates that the node
            cannot be uttered by the bot but still is a primary node. Which can be uttered by the user wihtout context.
        """
        return set(filter(lambda node: node.rating >= 0, self.z_nodes))

    @property
    def primary_z_labels(self) -> Set[Text]:
        return self._label_set_from_nodes(self.primary_z_nodes)

    def get_faq_list(self) -> List[Tuple[Text, Text]]:
        return [(question.full_text, answer.full_text) for question, answer in self.faq_node_pairs]

    def get_counter_args(self, label: Text) -> Optional[Set[GraphNode]]:
        """
        Returns all Responsenodes that directly counter the node with the label 'label' or None
        if none are found. Finds counterarguments for z, nz arguments and faqs.
        Args:
            label: Label of an Argument node e.g. "Z.P1" or "NZ.K1"
        """
        node = self.get_node_for_label(label)

        if node is None:
            return None

        if len(node.child_nodes) > 0:
            return node.child_nodes

        return None

    def get_node_for_label(self, label: Text) -> Optional[GraphNode]:
        """
        Searches for a label in the Collection no matter if z argument, nz argument, introduction, transition or faq.
        If ResponseTemplate for a label is found the node is returned, None otherwise.
        Args:
            label: Text
        Returns: ResponseTemplate corresponding to the parameter 'label' or None.

        """
        result = self._label_to_node.get(label, None)
        return result

    def get_num_arguments_pro(self) -> int:
        return len(
            set(
                filter(
                    lambda node: node.stance == Stance.PRO,
                    self.first_level_z_nodes,
                )
            )
        )

    def get_num_arguments_con(self) -> int:
        return len(
            set(
                filter(
                    lambda node: node.stance == Stance.CON,
                    self.first_level_z_nodes,
                )
            )
        )

    @staticmethod
    def _load_nodes_dataframes(nodes_directory_path: Text, file_extension: Text) -> Dict[Text, pd.DataFrame]:
        nodes_filenames = {
            "intros": "nodes_introductions",
            "transitions": "nodes_transitions",
            "groups": "nodes_groups",
            "zargs": "nodes_zargs",
            "nzargs": "nodes_nzargs",
            "faq": "nodes_faq",
        }
        nodes_filepaths = {
            name: os.path.join(nodes_directory_path, f"{filename}.{file_extension}")
            for name, filename in nodes_filenames.items()
        }
        nodes_dataframes = {
            name: load_nodes_df_from_file(filepath, file_extension)
            for name, filepath in nodes_filepaths.items()
        }
        return nodes_dataframes

    @staticmethod
    def _load_samples_dataframe(
            nodes_directory_path: str, samples_file_path: Optional[str], file_extension: Text
    ) -> Optional[pd.DataFrame]:
        if samples_file_path is None:
            samples_file_name = "samples_collection"
            samples_file_path = os.path.join(nodes_directory_path, f"{samples_file_name}.{file_extension}")

            if not os.path.exists(samples_file_path):
                return None

        return load_samples_df_from_file(samples_file_path, file_extension)

    @staticmethod
    def from_dataframes(
            nodes_dataframes: Dict[Text, pd.DataFrame],
            samples_dataframe: Optional[pd.DataFrame] = None,
    ) -> ArgumentGraph:
        nodes = {
            name: get_nodes_from_df(df, samples_dataframe)
            for name, df in nodes_dataframes.items()
        }

        return ArgumentGraph(
            introduction_nodes=nodes["intros"],
            transition_nodes=nodes["transitions"],
            z_arguments_nodes=nodes["zargs"],
            group_nodes=nodes["groups"],
            nz_arguments_nodes=nodes["nzargs"],
            faq_nodes=nodes["faq"],
        )

    @staticmethod
    def from_files(
            nodes_directory_path: Text,
            file_extension: Text,
            samples_file_path: Optional[Text] = None
    ) -> ArgumentGraph:
        nodes_dataframes = ArgumentGraph._load_nodes_dataframes(
            nodes_directory_path, file_extension
        )
        samples_dataframe = ArgumentGraph._load_samples_dataframe(
            nodes_directory_path, samples_file_path, file_extension
        )
        return ArgumentGraph.from_dataframes(nodes_dataframes, samples_dataframe)

    @staticmethod
    def from_csv_files(
            nodes_directory_path: Text,
            samples_file_path: Optional[Text] = None,
    ) -> ArgumentGraph:
        """
        Loads nodes from csv files.

        Args:
            nodes_directory_path: The folder where the node csv files are located.
            samples_file_path: If not None, the path to the samples csv file, otherwise the samples file is assumed to
            be located in the nodes_directory_path and the name is assumed to be "samples_collection.csv".
        Returns:

        """
        return ArgumentGraph.from_files(
            nodes_directory_path, "csv", samples_file_path)

    def _label_set_from_nodes(self, nodes: Iterable[GraphNode]) -> Set[Text]:
        return set(node.label for node in nodes)


if __name__ == "__main__":
    argument_graph = ArgumentGraph.from_csv_files("../argument_graphs/medai")
    print("done")
