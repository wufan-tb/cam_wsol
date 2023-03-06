# A Toy Weakly Supervised Object Localization using CAM

a toy weakly supervised object localization using cam(**Class Activation Mapping**, proposed in "Learning Deep Features for Discriminative Localization").

## Steps:

### 1. download classification dataset(dog vs cat) from [kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data?select=train.zip)

### 2. unzip dataset to "data/dog_vs_cat":

after unzip, folder structure should be:

````
```
data
└── dog_vs_cat
    ├── test
    │   ├── 1.jpg
    │   └── ...
    └── train
        ├── cat.0.jpg
        └── ...
```
````

### 3. traing

````
```
python train.py --data_path data/dog_vs_cat
```
````

### 4. inference

````
```
python inference.py --input {single image path or images folder}
```
````

## Demo:

![demo](demo.png)
