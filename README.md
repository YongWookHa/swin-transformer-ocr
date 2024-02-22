# swin-transformer-ocr
ocr with [swin-transformer](https://arxiv.org/abs/2103.14030)

## Overview
Simple and understandable swin-transformer OCR project.
The model in this repository heavily relied on high-level open-source projects like [timm](https://github.com/rwightman/pytorch-image-models) and [x_transformers](https://github.com/lucidrains/x-transformers).
And also you can find that the procedure of training is intuitive thanks to the legibility of [pytorch-lightning](https://www.pytorchlightning.ai/).

The model in this repository encodes input image to context vector with 'shifted-window` which is a swin-transformer encoding mechanism. And it decodes the vector with a normal auto-regressive transformer.

If you are not familiar with transformer OCR structure, [transformer-ocr](https://github.com/YongWookHa/transformer-ocr) would be easier to understand because it uses a traditional convolution network (ResNet-v2) for the encoder.
## Setup
```bash
conda create -n <env-name> python==3.7.7 && conda activate <env-name> && pip install basicsr
```
## Clone & Installation
```bash
git clone github.com/YongWookHa/swin-transformer-ocr.git && cd swin-transformer-ocr && mkdir checkpoints && pip install -r requirements.txt
```
## Performance
With private korean handwritten text dataset, the accuracy(exact match) is 97.6%.

## Data
```bash
./dataset/
├─ preprocessed_image/
│  ├─ cropped_image_0.jpg
│  ├─ cropped_image_1.jpg
│  ├─ ...
├─ train.txt
└─ val.txt

# in train.txt
cropped_image_0.jpg\tHello World.
cropped_image_1.jpg\tvision-transformer-ocr
...
```

You should preprocess the data first. Crop the image by word or sentence level area. Put all image data in a specific directory. Ground truth information should be provided with a txt file. In the txt file, write the image file name and label with `\t` separator in the same line.

## Configuration
In `settings/` directory, you can find `default.yaml`. You can set almost every hyper-parameter in that file. Copy one and edit it as your experiment version. I recommend you to run with the default setting first, before you change it.

## Train
```bash
python run.py --version 0 --setting settings/default.yaml --num_workers 16 --batch_size 128
```
You can check your training log with tensorboard.  
```
tensorboard --log_dir tb_logs --bind_all
```  

## Predict  
When your model finishes training, you can use your model for prediction.

```bash  
python predict.py --setting <your_setting.yaml> --target <image_or_directory> --tokenizer <your_tokenizer_pkl> --checkpoint <saved_checkpoint>
```

## Exporting to ONNX  
You can export your model to ONNX format. It's very easy thanks to pytorch-lightning. See the [related pytorch-lightning document](https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html).

## Citations

```bibtex
@misc{liu-2021,
    title   = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
	author  = {Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
	year    = {2021},
    eprint  = {2103.14030},
	archivePrefix = {arXiv}
}
```
