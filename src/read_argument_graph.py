import os
import re
import pandas as pd
from typing import Iterator, Optional, Text, List
from src.graph_node import GraphNodeCategory, GraphNode, Stance


def read_file(file_path: Text, file_format: Text, **read_config) -> pd.DataFrame:
    # if not os.path.exists(file_path):
    #     raise ValueError(f"file {file_path} does not exist")

    READ_CONFIG = {"keep_default_na": True}
    READ_CONFIG.update(read_config)

    if file_format in {"xlsx", "excel"}:
        df = pd.read_excel(file_path, **READ_CONFIG)
    elif file_format == "csv":
        df = pd.read_csv(file_path, skip_blank_lines=True, skipinitialspace=True, **READ_CONFIG)
    else:
        raise ValueError(f"unknown file format {file_format}")

    # strip all strings in dataframe
    df.map(lambda s: s.strip() if isinstance(s, str) else s)

    return df


def read_graph_nodes_file(file_path: Text, file_format: Text) -> pd.DataFrame:
    TEMPLATE_DTYPES = {
        "label": str,
        "category": str,
        "stance": str,
        "parents": str,
        "similars": str,
        "label_name": str,
        "paraphrase": str,
        "summary": str,
        "full_text": str,
        "groups": str,
        "rating": pd.Int64Dtype(),
        "reaction_to_argument": pd.Int64Dtype(),
    }

    READ_CONFIG = {
        "dtype": TEMPLATE_DTYPES,
        "usecols": TEMPLATE_DTYPES.keys(),
    }

    return read_file(file_path, file_format, **READ_CONFIG)


def read_samples_file(file_path: Text, file_format: Text) -> pd.DataFrame:
    SAMPLES_COLLECTION_DTYPES = {
        "label": str,
        "text": str,
    }

    READ_CONFIG = {
        "dtype": SAMPLES_COLLECTION_DTYPES,
        "usecols": SAMPLES_COLLECTION_DTYPES.keys(),
    }

    return read_file(file_path, file_format, **READ_CONFIG)


def read_nodes_with_previous_label_file(file_path: Text, file_format: Text) -> pd.DataFrame:
    TEMPLATE_ZARGS_DTYPES = {
        "previous_label": str,
        "label": str,
        "category": str,
        "stance": str,
        "parents": str,
        "similars": str,
        "label_name": str,
        "paraphrase": str,
        "summary": str,
        "full_text": str,
        "groups": str,
        "rating": pd.Int64Dtype(),
        "reaction_to_argument": pd.Int64Dtype(),
    }

    READ_CONFIG = {
        "dtype": TEMPLATE_ZARGS_DTYPES,
        "usecols": TEMPLATE_ZARGS_DTYPES.keys(),
    }

    return read_file(file_path, file_format, **READ_CONFIG)


def load_nodes_df_from_file(file_path: Text, file_format: Text) -> pd.DataFrame:
    print(f"Loading nodes from file {file_path} ... ", end="")
    df = read_graph_nodes_file(file_path, file_format)
    print("Done")
    return df


def load_samples_df_from_file(file_path: Text, file_format: Text) -> pd.DataFrame:
    print(f"Loading samples from file {file_path} ... ", end="")
    df = read_samples_file(file_path, file_format)
    print("Done")
    return df


def split_by_comma(text: Text) -> Iterator[Text]:
    match_iter = re.finditer(r"[^,\s]+", text)
    string_iter = (text[match.start(): match.end()] for match in match_iter)
    return string_iter


def parse_samples_from_dataframe(label: Text, samples_dataframe: pd.DataFrame) -> Iterator[Text]:
    for _, row in samples_dataframe.iterrows():
        if row["label"] == label:
            yield str(row["text"])


def parse_parents_string(parents_string: Text) -> Iterator[Text]:
    return split_by_comma(parents_string)


def parse_similars_string(similars_string: Text) -> Iterator[Text]:
    return split_by_comma(similars_string)


def parse_groups_string(groups_string: Text) -> Iterator[Text]:
    return split_by_comma(groups_string)


def parse_category_string(category_string: Text) -> GraphNodeCategory:
    return GraphNodeCategory.from_string(category_string)


def parse_stance_string(stance_string: Text) -> Stance:
    return Stance.from_string(stance_string)


def get_nodes_from_df(
    graph_nodes_df: pd.DataFrame,
    samples_dataframe: Optional[pd.DataFrame] = None,
) -> List[GraphNode]:
    """
    Takes a dataframe containing all nodes in predefined table format.
    Empty rows will be ignored.
    Templates without label or full_text will be ignored.

    Parameters
    ----------
    graph_nodes_df: DataFrame
        the pandas dateframe containing the graph_nodes as table

    samples_dataframe: DataFrame
        the pandas dataframe containing text samples for the graph_nodes

    Returns
    -------
    List[GraphNode]
        List of Graph Nodes from the given dataframe.
    """
    graph_nodes = []
    for _, row in graph_nodes_df.iterrows():
        # create init dict for node with only available keys
        args = {k: v for k, v in row.items() if not pd.isna(v) and k != "Unnamed: 0"}

        # check minimum requirements to init response node
        if "label" not in args or "full_text" not in args:
            continue

        # if samples dataframe is available parse the samples for the label
        if samples_dataframe is not None:
            args["samples"] = set(parse_samples_from_dataframe(args["label"], samples_dataframe))

        # parse category string to Category
        if "category" in args:
            args["category"] = parse_category_string(args["category"])

        # parse stance string to Stance
        if "stance" in args:
            args["stance"] = parse_stance_string(args["stance"])

        # parse parents string to Set of labels
        if "parents" in args:
            args["parents"] = set(parse_parents_string(args["parents"]))

        # parse similars string to Set of labels
        if "similars" in args:
            args["similars"] = set(parse_parents_string(args["similars"]))

        # parse groups string to Set of labels
        if "groups" in args:
            args["groups"] = set(parse_groups_string(args["groups"]))

        graph_nodes.append(GraphNode(**args))

    return graph_nodes


if __name__ == "__main__":
    print(os.getcwd())
    nodes = get_nodes_from_df("../argument_graphs/szenario_medai")
    # response_nodes = get_nodes('./szenario_3/nodes_nzargs_s3.xlsx')
    # response_nodes = get_nodes('./response_nodes/szenario3/nodes_zargs_s3.xlsx')
    # response_nodes = get_nodes('./szenario_3/nodes_transition_s3.xlsx')
    # response_nodes = get_nodes_szenario_2(
    #     "../../data/input/response_nodes/szenario2/nodes_szenario2_s2old.xlsx")
    # transitions_s2 = get_intro_trans_szenario_2(
    #     "../../data/input/response_nodes/szenario2/nodes_transition_s2old.xlsx")
    # introductions_s2 = get_intro_trans_szenario_2(
    #     "../../data/input/response_nodes/szenario2/nodes_introduction_s2old.xlsx")
    # print(os.getcwd())
    # # print(response_nodes)
    # for node in response_nodes:
    #     print(node)
    # tmp_var = [(tmp.label, tmp.kurzfassung, tmp.schablone) for tmp in response_nodes]
    # labels, shorts, nodes = list(zip(*tmp_var))
    # print(labels)
    # print(shorts)
    # print(nodes)

    # from model.response_node_collection import ResponseTemplateCollection

    # rtc = ResponseTemplateCollection.from_csv_files("response_nodes/szenario_s1")
    # ResponseTemplateCollection.from_xlsx_files("response_nodes/szenario3")

    print("pass")
