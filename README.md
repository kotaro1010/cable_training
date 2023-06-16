# 斜張橋ケーブル異常検知第三次開発

## Setup
```bash
cd docker && bash build.sh
bash run.sh
```

## Inference
1. ```python3 predict.py```

## Train
1. datasetの用意
    - 各ラベルごとの画像が格納されたフォルダを用意。
1. ```python3 split_dataset.py```
1. ```python3 train.py```
