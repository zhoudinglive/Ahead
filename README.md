# Ahead
This is the source code of KDD21 paper "Attentive Heterogeneous Graph Embedding for Job Mobility Prediction", where the file structure is shown below. Although we cannot publish our dataset for the security problem, you can find the input&output format of our model in source_data/dataset.py. Feel free to contact us if you have any issues.

```
.
├── LICENSE
├── README.md
├── main.py
├── model
│   ├── ahead.py
│   ├── config.py
│   ├── dgru.py
│   ├── hgcn.py
│   └── utils.py
└── source_data
    └── dataset.py
```

If you find Ahead useful in your research, we ask that you cite the following paper

```
@inproceedings{zhang2021attentive,
  title={Attentive heterogeneous graph embedding for job mobility prediction},
  author={Zhang, Le and Zhou, Ding and Zhu, Hengshu and Xu, Tong and Zha, Rui and Chen, Enhong and Xiong, Hui},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={2192--2201},
  year={2021}
}
```
