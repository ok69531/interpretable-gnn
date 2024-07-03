# Interpretable GNN

## Information Bottleneck
* Graph Information Bottleneck for Subgraph Recognition, arXiv, 2020 [[pdf]](https://arxiv.org/pdf/2010.05563), [[implementation]](./gib/)
* Improving Subgraph Recognition with Variational Graph Information Bottleneck, CVPR, 2022 [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Improving_Subgraph_Recognition_With_Variational_Graph_Information_Bottleneck_CVPR_2022_paper.pdf), [[implementation]](./vgib/)
* Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism, ICML, 2022 [[pdf]](https://arxiv.org/pdf/2201.12987), [[implementation]](./gsat/)
* Interpretable Prototype-based Graph Information Bottleneck, Neurips, 2024 [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2023/file/f224f056694bcfe465c5d84579785761-Paper-Conference.pdf), [[implementation]](./pgib/)

## Implementation Results
Backbone model: GIN

**Graph Calssification - Accuracy**

| Dataset  |    GIB     |    VGIB    |   GSAT    |   PGIB    |   PGIB<sub>cont</sub>    |
| -------  |     ---    |    ----    |   ----    |   ----    |   ----    |
| MUTAG    | ${0.790}_{\pm 0.054}$ | ${0.760}_{\pm 0.044}$ | ${0.805}_{\pm 0.091}$ | ${0.805}_{\pm 0.076}$ | ${0.825}_{\pm 0.060}$ |
| PROTEINS | ${0.741}_{\pm 0.029}$ | ${0.755}_{\pm 0.027}$ | ${0.740}_{\pm 0.016}$ | ${0.638}_{\pm 0.024}$ | ${}_{\pm }$ |
| NCI1     | ${0.702}_{\pm 0.022}$ | ${0.644}_{\pm 0.019}$ | ${0.740}_{\pm 0.016}$ | - | ${0.736}_{\pm 0.014}$ |
| DD       | ${0.728}_{\pm 0.027}$ | ${0.728}_{\pm 0.057}$ | ${0.694}_{\pm 0.020}$ | - | ${}_{\pm }$ |
| IMDB-B   | ${0.698}_{\pm 0.022}$ | ${0.657}_{\pm 0.018}$ | ${0.716}_{\pm 0.021}$ | - | ${0.595}_{\pm 0.049}$ |
| IMDB-M   | ${0.376}_{\pm 0.035}$ | ${0.343}_{\pm 0.037}$ | ${0.359}_{\pm 0.031}$ | - | ${0.336}_{\pm 0.020}$ |


**Graph Interpretation - Fidelity**

| Dataset  |    GIB     |    VGIB    |   GSAT    |   PGIB    |   PGIB<sub>cont</sub>    |
| -------  |     ---    |    ----    |   ----    |   ----    |   ----    |
| QED      |            |            |           |           |           |
| DRD2     |            |            |           |           |           |
| HLM      |            |            |           |           |           |
| MLM      |            |            |           |           |           |
| RLM      |            |            |           |           |           |

