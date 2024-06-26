# Interpretable GNN

## Information Bottleneck
* Graph Information Bottleneck for Subgraph Recognition, arXiv, 2020 [[pdf]](https://arxiv.org/pdf/2010.05563), [[implementation]](./gib/)
* Improving Subgraph Recognition with Variational Graph Information Bottleneck, CVPR, 2022 [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Improving_Subgraph_Recognition_With_Variational_Graph_Information_Bottleneck_CVPR_2022_paper.pdf), [[implementation]](./vgib/)
* Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism, ICML, 2022 [[pdf]](https://arxiv.org/pdf/2201.12987), [implementation]
* Interpretable Prototype-based Graph Information Bottleneck, Neurips, 2024 [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2023/file/f224f056694bcfe465c5d84579785761-Paper-Conference.pdf), [[implementation]](./pgib/)

## Implementation Results
Backbone model: GIN

<!-- PGIB lr for MUTAG: 0.001 \
PGIB lr for PROTEINS: 0.005 -->

**Graph Calssification**

| Dataset  |    GIB     |    VGIB    |   GSAT    |   PGIB    |
| -------  |     ---    |    ----    |   ----    |   ----    |
| MUTAG    | ${0.79}_{\pm 0.054}$ | ${0.760}_{\pm 0.044}$ | ${}_{\pm }$ | ${}_{\pm }$ |
| PROTEINS | ${0.741}_{\pm 0.029}$ | ${0.755}_{\pm 0.027}$ | ${}_{\pm }$ | ${}_{\pm }$ |
| NCI1     | ${0.702}_{\pm 0.022}$ | ${0.644}_{\pm 0.019}$ | ${}_{\pm }$ | ${}_{\pm }$ |
| DD       | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |
| IMDB-B   | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |
| IMDB-M   | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | 


**Graph Interpretation**

| Dataset  |    GIB     |    VGIB    |   GSAT    |   PGIB    |
| -------  |     ---    |    ----    |   ----    |   ----    |
| QED | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |
| DRD2 | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |
| HLM | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |
| MLM | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |
| RLM | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ | ${}_{\pm }$ |

