# VisualDomainDecathlon

### Prerequisites

Install the following:

```
- Python >= 3.6
- pytorch >= 1.0.1
- torchvision >= 0.2.2
```

Then, install python packages:

```
pip install -r requirements.txt
```

### Datasets

#####Download

Download [Visual Domain Decathlon](https://www.robots.ox.ac.uk/~vgg/decathlon/#download) Datasets except(ImageNet), [Caltech256, CIFAR-10, Sketches](www.google.com) and move 12 datasets to **$DIR/decathlon-1.0-data**.

We have rearranged the format of all datasets to suit torchvision.

##### Dataset Shrink

```
cd $DIR/decathlon-1.0-data
# 1/10 size
python dataset_shrink.py --save-dir decathlon-1.0-data-tenth --shrink-ratio 10
# 1/100 size
python dataset_shrink.py --save-dir decathlon-1.0-data-hundredth --shrink-ratio 100
```

### Training

##### Baseline

```
cd $DIR/src
python decathlon_baseline.py --data-dir decathlon-1.0-data --log-dir log_save --model-save-dir model_weights --depth 28 --widen-factor 1 
```

### Finetune

| Train Dataset | Finetune |                           Command                            |
| :-----------: | :------: | :----------------------------------------------------------: |
|     100%      | Fc layer | python decathlon_transfer.py --data-dir decathlon-1.0-data --transfer-result-dir transfer_result_fc_all --fc |
|     100%      |   All    | python decathlon_transfer.py --data-dir decathlon-1.0-data --transfer-result-dir transfer_result_all |
|      10%      | Fc layer | python decathlon_transfer.py --data-dir decathlon-1.0-data-tenth --transfer-result-dir transfer_result_fc_all_tenth --fc |
|      10%      |   All    | python decathlon_transfer.py --data-dir decathlon-1.0-data-tenth --transfer-result-dir transfer_result_all_tenth --fc |

### Acknowledgement

We thank [Mtan](https://github.com/lorenmt/mtan) for providing some source codes.

### Contact

chenyix@zju.edu.cn





