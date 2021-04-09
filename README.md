# fraud_detection
AE for fraud_detection 

# how to run

1. make `./data` dir 

2. [load data](https://www.kaggle.com/jacklizhi/creditcard) to `./data` dir

3. run `python trainer.py`


```
base ❯ python trainer.py --help      
usage: trainer.py [-h] [--seed SEED] [--device DEVICE] [--model MODEL]
                  [--data_name DATA_NAME] [--batch_size BATCH_SIZE]
                  [--epochs EPOCHS] [--optimizer OPTIMIZER] [--lr LR]
                  [--val_iter VAL_ITER]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  --device DEVICE
  --model MODEL
  --data_name DATA_NAME
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --optimizer OPTIMIZER
  --lr LR
  --val_iter VAL_ITER
```
