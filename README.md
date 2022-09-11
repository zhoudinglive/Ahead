# Ahead
This is the source code of KDD21 paper "Attentive Heterogeneous Graph Embedding for Job Mobility Prediction", where the file structure is shown below. Although we cannot publish our dataset for the security problem, you can find the input&output format of our model in source_data/dataset.py. Feel free to contact us if you have any issues.
- file structure
```
.
├── LICENSE
├── README.md
├── main.py
├── model
│   ├── __pycache__
│   │   ├── ahead.cpython-38.pyc
│   │   ├── config.cpython-38.pyc
│   │   ├── dgru.cpython-38.pyc
│   │   ├── hgcn.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── ahead.py
│   ├── config.py
│   ├── dgru.py
│   ├── hgcn.py
│   └── utils.py
├── model_save
│   └── ahead_duration
├── requirements.txt
└── source_data
    ├── dataset.py
    ├── mock_company_ppr.npy
    ├── mock_dur_of_company.npy
    ├── mock_dur_of_title.npy
    ├── mock_external_graph.npy
    ├── mock_internal_graph.npy
    ├── mock_pid_str_dict.json
    ├── mock_final_records.npy
    ├── mock_skill_embedding.npy
    └── mock_title_ppr.npy
```
- code running
```commandline
pip install requirements.txt
python main.py
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
