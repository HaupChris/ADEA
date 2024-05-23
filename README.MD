# ADEA: An Argumentative Dialog Dataset on Ethical Issues concerning Future A.I. Applications
## Overview
This repository contains the data and code for the paper [ADEA: An Argumentative Dialog Dataset on Ethical Issues concerning Future A.I. Application](https://aclanthology.org/2024.lrec-main.38/) accepted at the 2024 LREC-COLING.

# Structure
The repository is organized into three main folders:
- `dialogues/`: contains the Dialogue dataset in csv format. For each topic of discussion there are four files. annotator_1.csv and annotator_2.csv contain the dialogues annotated by the two annotators. The file annotation_final.csv contains the final annotations of the dialogues, which were obtained by resolving the disagreements between the two annotators by a third annotator. The file predictions_gpt4.csv contains the predictions of the OpenAI GPT-4 model for the benchmark, described in the paper.
- `argument_graphs/`: Contains the argument graph for each topic of discussion in the dataset. The argument graphs are in csv format. Each directory holds the following files:
    - `nodes_zargs.csv`: contains the nodes of the argument graph for well-founded arguments.
    - `nodes_nzargs.csv`: contains the nodes of the argument graph for a selection of unfounded-founded arguments.
    - `nodes_groups.csv`: contains the nodes for argument groups.
    - `nodes_faq.csv`: contains the nodes for frequently asked questions.
    - `nodes_introductions.csv` contains nodes for introductary phrases, utilized by the dialogue system that was used to collect the dialogues.
    - `nodes_transitions.csv` contains nodes for transition phrases, utilized by the dialogue system that was used to collect the dialogues.
    - Additionally argument samples can be added to the graph, e.g. different textual version of a certain argument. This can be done by adding a file `samples_collection.csv` to the directory. This file should contain the header `id, label, text` and the respective samples. id can be an arbitrary number, label should be the label of the respective argument and text can be for example from a dialogue or a different source. An example file can be found in the directory `argument_graphs/medai/`.
- `src/`: contains the code to work with the argument graphs.


## Installation and Usage
The repository requires Python 3.10. To install the required packages, run:
```bash
pip install -r requirements.txt
```

To work with the argument graphs, you can use the code the `src/arugment_graph.py`. The following code snippet shows how to load an argument graph and print the nodes of the graph:
```python
from src.argument_graph import ArgumentGraph

# Load the argument graph
graph = ArgumentGraph("argument_graphs/medai/")
# Print texts of the one node of well-founded arguments
print(graph.nodes_zargs[0].full_text)
```

## Contributing
If you would like to contribute to the project, please open an issue or a pull request. We are happy to receive any feedback or suggestions.


## Referencing
If you use this project inyour research or wish to refer to the baseline results, please use the following citation: 

```
@inproceedings{hauptmann2024adea,
  title={ADEA: An Argumentative Dialogue Dataset on Ethical Issues Concerning Future AI Applications},
  author={Hauptmann, Christian and Krenzer, Adrian and Krause, Antonia and Puppe, Frank},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={426--437},
  year={2024}
}
```

## License

This project and its content are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). This allows for both personal and commercial use, distribution, adaptation, and building upon the work, as long as appropriate credit is given and any new creations are licensed under identical terms.

For more details, see the full license [here](http://creativecommons.org/licenses/by-sa/4.0/).


