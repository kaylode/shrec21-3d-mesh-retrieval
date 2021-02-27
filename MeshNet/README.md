# MeshNet

<img src="./doc/pipeline.png" width="700">

## Folder Structure
```
this repo
│   test.py
│   extract_features.py
│   train.py
│
└───datasets  
│   └───datasetCulture
│       └───annotations
│       │     dataset.csv
│       │     test.csv
│       │     0_train.csv
│       │     0_val.csv
│       │     ....
│       └───objects
│           └─── train
|           |    0.obj
|           |    ...
│           └─── test
|           |    ...
│   └───datasetShape
│       ....  
```

## Dataset
- Download original dataset: [link](https://drive.google.com/file/d/11GUD6EiKN-MMqGeNT8wI7ibpVfaRFC4w/view?usp=sharing)

## Pipeline
- Download and make dataset as above
- Change configs in all shell script files in ```scripts/```. Examples:
  ```
  ROOT='/home/pmkhoi/shrec21/retrieval/MeshNet/'
  FOLD=4
  TASK='Shape'
  NUM_FACES=20000
  ```
- Run ```prepare_data.sh``` to simplify and generate dataset
- Run ```train.sh``` to start training, training configs can be found in ```config/train_config.yaml``
- Specify WEIGHT in ```eval.sh``` then run to calculate metric scores
- Run ```submission.sh``` to generate distance matrix for submission

## Paper References:
- [MeshNet: Mesh Neural Network for 3D Shape Representation](http://gaoyue.org/paper/MeshNet.pdf)

## Code References:
- HCMUS-Team
- https://github.com/iMoonLab/MeshNet
