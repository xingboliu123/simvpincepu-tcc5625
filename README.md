# PhySimCast for Weather Forecasting

This repository contains the implementation code for PhySimCast on the WeatherBench dataset.

## 1. Environment Installation

This project uses the same environment settings as OpenSTL.

```shell
git clone [https://github.com/xingboliu123/physimcast.git](https://github.com/xingboliu123/physimcast.git)
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

## 2. Dataset Preparation

We use the **WeatherBench** dataset for training and evaluation.

* **Dataset Link:** [WeatherBench Official Repository](https://github.com/pangeo-data/WeatherBench)
* **Download:** Please download the required data and place it in the `./data` directory.

## 3. Usage

### Training

To train the model on WeatherBench:

```shell
python tools/train.py     --dataname weather_tcc_5_625       --data_root your     --config_file configs/weather/tcc_5_625/SimVP_IncepU.py     --ex_name weather_tcc_5_625/SimVP_IncepU
python tools/train.py     --dataname weather_tcc_2_8125       --data_root your     --config_file configs/weather/tcc_5_625/SimVP_IncepU.py     --ex_name weather_tcc_2_8125/SimVP_IncepU
python tools/train.py     --dataname weather_tcc_1_40625       --data_root your     --config_file configs/weather/tcc_5_625/SimVP_IncepU.py     --ex_name weather_tcc_1_40625/SimVP_IncepU
```

### Testing

To test the pre-trained model:

```shell
python tools/test.py  --dataname weather_tcc_5_625   --config_file configs/weather/tcc_5_625/SimVP_IncepU.py   --ckpt_path  your ckpt   --data_root   your data_root
python tools/test.py  --dataname weather_tcc_2_8125   --config_file configs/weather/tcc_5_625/SimVP_IncepU.py   --ckpt_path  your ckpt   --data_root   your data_root
python tools/test.py  --dataname weather_tcc_1_40625   --config_file configs/weather/tcc_5_625/SimVP_IncepU.py   --ckpt_path  your ckpt   --data_root   your data_root
```
