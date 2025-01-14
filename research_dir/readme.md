# Enhancing Explainability in Graph Neural Networks Using Game-Theoretic Approaches

## Overview

Graph Neural Networks (GNNs) have become essential tools for modeling complex graph-structured data, but their opaque decision-making processes pose significant challenges for interpretability. This project introduces two game-theoretic approaches to enhance the explainability of GNNs:

1. **Approximate Shapley Value Computation**: An efficient method for approximating Shapley values using Monte Carlo sampling, which substantially reduces computational complexity while maintaining high explanatory fidelity.

2. **Structure-Aware Explanations Using the Hamiache-Navarro (HN) Value**: A technique that incorporates graph topology into the interpretation process, allowing for a nuanced assessment of node importance by considering both feature contributions and structural roles.

We integrate these methods into existing GNN frameworks and evaluate them on benchmark datasets, including citation networks and molecular graphs. Our experimental results demonstrate improved explanation fidelity and computational efficiency, advancing the development of more transparent and interpretable graph-based models.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Experiments](#experiments)
  - [Experiment 1: Approximate Shapley Values](#experiment-1-approximate-shapley-values)
  - [Experiment 2: Structure-Aware Explanations Using HN Value](#experiment-2-structure-aware-explanations-using-hn-value)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- NumPy
- SciPy
- scikit-learn
- NetworkX
- Matplotlib

You can install the required packages using:

```bash
pip install -r requirements.txt
```

*Note: Ensure that you have the appropriate version of Python installed. You may consider using a virtual environment to manage dependencies.*

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/gnn-explainability.git
cd gnn-explainability
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data Preparation

### SST-2 Dataset

For Experiment 1, we use the Stanford Sentiment Treebank (SST-2) dataset. Download the dataset from the [official website](https://nlp.stanford.edu/sentiment/index.html) or using the [GLUE benchmark](https://gluebenchmark.com/tasks).

Place the SST-2 data in the `data/sst2/` directory:

```
data/
└── sst2/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
```

### MUTAG Dataset

For Experiment 2, we use the MUTAG dataset. Download the dataset from the [TU Dortmund University](https://chrsmrrs.github.io/datasets/docs/datasets/) collection.

Place the MUTAG data in the `data/mutag/` directory:

```
data/
└── mutag/
    ├── MUTAG_A.txt
    ├── MUTAG_graph_labels.txt
    ├── MUTAG_graph_indicator.txt
    ├── MUTAG_node_labels.txt
    └── MUTAG_edge_labels.txt
```

## Experiments

### Experiment 1: Approximate Shapley Values

In this experiment, we develop an efficient approximation method for computing Shapley values in GNN explanations using Monte Carlo sampling.

#### Steps:

1. **Data Loading and Preprocessing**:
   - Load a subset of the SST-2 dataset.
   - Preprocess sentences by tokenization and TF-IDF vectorization.
   - Construct a $k$-nearest neighbor graph based on cosine similarity with $k=5$.

2. **Model Training**:
   - Implement a two-layer Graph Convolutional Network (GCN) without external libraries.
   - Train the GCN on the constructed graph.

3. **Approximate Shapley Value Computation**:
   - Set the number of Monte Carlo samples `M = 50`.
   - For each node, approximate its Shapley value using Monte Carlo sampling.

4. **Visualization**:
   - Plot the approximate Shapley values to visualize nodes' feature importance.

#### Run the Experiment:

```bash
python experiment1_shapley.py
```

### Experiment 2: Structure-Aware Explanations Using HN Value

In this experiment, we propose a structure-aware GNN explanation method based on the Hamiache-Navarro (HN) value.

#### Steps:

1. **Data Loading**:
   - Load the MUTAG dataset, which consists of molecular graphs.

2. **Model Training**:
   - Implement and train a GCN on the MUTAG data.

3. **HN Value Computation**:
   - Calculate the HN values for nodes based on their degrees and the graph's topology.

4. **Visualization**:
   - Plot the HN values to visualize nodes' structural importance.

#### Run the Experiment:

```bash
python experiment2_hn.py
```

## Results

Results from the experiments, including plots and computed values, are saved in the `results/` directory.

- **Experiment 1**:
  - Approximate Shapley values are saved as `results/experiment1_shapley_values.png`.
  - The distribution of Shapley values indicates nodes with higher feature importance.

- **Experiment 2**:
  - HN values are saved as `results/experiment2_hn_values.png`.
  - The HN values highlight structurally important nodes in the graph.

## Usage

### Reproducing the Experiments

1. **Clone the Repository and Install Dependencies**:

   ```bash
   git clone https://github.com/your_username/gnn-explainability.git
   cd gnn-explainability
   pip install -r requirements.txt
   ```

2. **Prepare the Datasets**:

   - Download the SST-2 and MUTAG datasets as described in [Data Preparation](#data-preparation).

3. **Run Experiment 1**:

   ```bash
   python experiment1_shapley.py
   ```

4. **Run Experiment 2**:

   ```bash
   python experiment2_hn.py
   ```

### Modifying Parameters

You can adjust parameters like the number of Monte Carlo samples `M` or the value of `k` in the k-nearest neighbor graph by editing the respective scripts.

For example, to change the number of Monte Carlo samples in `experiment1_shapley.py`:

```python
M = 100  # Increase the number of samples for higher accuracy
```

## Citation

If you use this code in your research, please cite our work:

```
@article{YourName2023Explainability,
  title={Research Report: Enhancing Explainability in Graph Neural Networks Using Game-Theoretic Approaches},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2023},
  volume={},
  number={},
  pages={}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Note: Replace `your_username`, `Your Name`, and other placeholders with the actual information relevant to your repository and publication.*