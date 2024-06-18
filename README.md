# Interpretable GNN

## Information Bottleneck
* Graph Information Bottleneck for Subgraph Recognition, arXiv, 2020 [[pdf]](https://arxiv.org/pdf/2010.05563), [[implementation]](./gib/)
* Improving Subgraph Recognition with Variational Graph Information Bottleneck, CVPR, 2022 [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Improving_Subgraph_Recognition_With_Variational_Graph_Information_Bottleneck_CVPR_2022_paper.pdf), [[implementation]](./vgib/)
* Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism, ICML, 2022 [[pdf]](https://arxiv.org/pdf/2201.12987), [[implementation]](./pgib/)
* Interpretable Prototype-based Graph Information Bottleneck, Neurips, 2024 [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2023/file/f224f056694bcfe465c5d84579785761-Paper-Conference.pdf), [implementation]

## Implementation Results
Backbone model: GIN

<!-- model initialization seed: 42, data split seed: 12345
| Dataset | GIB | VGIB | GSAT | PGIB |
| ------- | --- | ---- | ---- | ---- |
| MUTAG | ${0.792}_{\pm 0.100}$ | ${0.777}_{\pm 0.099}$ | ${}_{\pm }$ | ${}_{\pm }$ |
| PROTEINS | ${0.700}_{\pm 0.025}$ | ${0.634}_{\pm 0.058}$ | ${}_{\pm }$ | ${}_{\pm }$ | -->

model initialization seed: 0, data split seed: 42

PGIB lr for MUTAG: 0.001 \
PGIB lr for PROTEINS: 0.005
| Dataset | GIB | VGIB | GSAT | PGIB |
| ------- | --- | ---- | ---- | ---- |
| MUTAG | ${0.798}_{\pm 0.102}$ | ${0.765}_{\pm 0.123}$ | ${}_{\pm }$ | ${0.804}_{\pm 0.078}$ |
| PROTEINS | ${0.700}_{\pm 0.036}$ | ${0.660}_{\pm 0.065}$ | ${}_{\pm }$ | ${0.669}_{\pm 0.049}$ |
