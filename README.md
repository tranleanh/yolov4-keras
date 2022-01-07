## YOLOv4 Implementation in Tensorflow-Keras
---

This repo has been developed based on [this project](https://github.com/bubbliiiing/yolov4-keras). 

I have made an English version for those who got struggle with Chinese (including me).

## 0. Contents
1. [Performance](#1. Performance)
2. [Achievement](#Achievement)
3. [Environment](#Environment)
4. [Note](#Note)
5. [TrainingSettings](#TrainingSettings)
6. [Download](#Download)
7. [How2Train](#How2Train)
8. [How2Predict](#How2Predict)
9. [How2Eval](#How2Eval)
10. [Reference](#Reference)

## 1. Performance
| Trainset | Weight File | Testset | Input Size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolo4_voc_weights.h5](https://github.com/bubbliiiing/yolov4-keras/releases/download/v1.0/yolo4_voc_weights.h5) | VOC-Test07 | 416x416 | - | 88.9
| COCO-Train2017 | [yolo4_weight.h5](https://github.com/bubbliiiing/yolov4-keras/releases/download/v1.0/yolo4_weight.h5) | COCO-Val2017 | 416x416 | 46.4 | 70.5

## 2. Achievement
- [x] Backbone：DarkNet53 => CSPDarkNet53
- [x] Neck：SPP,PAN
- [x] Training Setting：Mosaic Data Augmentation, Label Smoothing、CIOU、Cosine Annealing Learning Rate Scheduler
- [x] Activation：Mish

## 3. Environment
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 4. Note
The weight file `yolo4_weights.h5` was trained with the anchors of input size 608x608.    
Be careful not to use Chinese labels, and no spaces in the folder!    
Before training, you need to create a new txt document under `model_data`, enter the classes to be classified in the document, and point `classes_path` to this file in `train.py`.

## 5. TrainingSettings
In `train.py`：   
1. mosaic: whether or not to use mosaic data augmentation
2. cosine_scheduler: whether or not to use cosine annealing learning rate scheduler   
3. label_smoothing: whether or not to use label smoothing 

## 6. Download
The weight file `yolo4_weights.h5` can be downloaded from [Baidu](https://pan.baidu.com/s/1R4LlPqVBdusVa9Mx_BXSTg). (Code: k8v5)

yolo4_weights.h5: weights trained on COCO dataset 

yolo4_voc_weights.h5: weights trained on VOC dataset 

Link to download VOC dataset: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A (Code: uack)   

## 7. How2Train
### a. Train on VOC07+12 dataset
1. Data preparation    
This project uses VOC data format for training. Before training, download the VOC07+12 data set, decompress it and place it in the root directory    

2. Data preprocessing   
Modify `annotation_mode=2` in `voc_annotation.py`, run `voc_annotation.py` to generate `2007_train.txt` and `2007_val.txt` in the root directory   

3. Start training   
The default parameters of `train.py` are used to train the VOC dataset, and you can start training directly by running `train.py`.   

4. Prediction   
Two files are needed to predict the training results, namely `yolo.py` and `predict.py`. We first need to modify `model_path` and `classes_path` in `yolo.py`, these two parameters must be modified.    
`model_path` points to the trained weight file, in the logs folder    
`classes_path` points to the txt corresponding to the detection category     
After the modification is completed, you can run `predict.py` for detection. After running, enter the image path to detect    

### b. Train your own data set    
1. Data preparation   
This article uses the VOC format for training, and you need to make a good data set before training    
Before training, put the label file in the Annotation under the VOC2007 folder under the VOCdevkit folder      
Before training, place the picture files in JPEGImages under the VOC2007 folder under the VOCdevkit folder    

2. Data preprocessing    
After finishing the placement of the data set, we need to use `voc_annotation.py` to obtain `2007_train.txt` and `2007_val.txt` for training       
Modify the parameters in `voc_annotation.py`. In the first training, only `classes_path` can be modified, and `classes_path` is used to point to the txt corresponding to the detection category       
When training your own data set, you can create a `cls_classes.txt`, which contains the categories you need to distinguish    
The content of the `model_data/cls_classes.txt` file is:  
```python
cat
dog
...
```
Modify the `classes_path` in `voc_annotation.py` to correspond to `cls_classes.txt`, and run `voc_annotation.py`.

3. Start training   
There are many training parameters, all of which are in `train.py`. You can read the comments carefully after downloading the library. The most important part is still the `classes_path` in `train.py`        
`classes_path` is used to point to the txt corresponding to the detection category, this txt is the same as the txt in `voc_annotation.py`! The data set for training yourself must be modified!    
After modifying the `classes_path`, you can run `train.py` to start training. After training multiple epochs, the weights will be generated in the logs folder     

4. Prediction  
Two files are needed to predict the training results, namely `yolo.py` and `predict.py`. Modify `model_path` and `classes_path` in `yolo.py`    
`model_path` points to the trained weight file, in the logs folder    
`classes_path` points to the txt corresponding to the detection category    
After the modification is completed, you can run `predict.py` for detection. After running, enter the image path to detect.    

## 8. How2Predict
### a. Use pre-trained weights
1. After downloading the repo, unzip it, download `yolo_weights.pth` from Baidu and put it in model_data, run `predict.py`:
```python
img/street.jpg
```
2. Setting in `predict.py` can perform fps test and video video detection       
### b. Use self-trained weights
1. Follow the training steps      
2. In the `yolo.py` file, modify `model_path` and `classes_path` in the following parts to make them correspond to the trained files; `model_path` corresponds to the weight file under the logs folder, and `classes_path` is the class corresponding to `model_path`      
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Use your own trained model to make predictions, you must modify model_path and classes_path!
    #   model_path points to the weight file under the logs folder, classes_path points to the txt under model_data
    #   If there is a shape mismatch, pay attention to the modification of the model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolo4_weight.h5',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
    #   anchors_mask is used to help the code find the corresponding a priori box, generally not modified.
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   The size of the input image must be a multiple of 32.
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   Only prediction boxes with a score greater than the confidence level will be retained
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   The size of nms_iou used for non-maximum suppression
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   Maximum number of boxes
    #---------------------------------------------------------------------#
    "max_boxes"         : 100,
    #---------------------------------------------------------------------#
    #   This variable is used to control whether to use letterbox_image to resize the input image without distortion
    #   After many tests, it is found that closing letterbox_image directly resizes better
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
}
```
3. Run `predict.py`, enter      
```python
img/street.jpg
```
4. Setting in `predict.py` can perform fps test and video video detection.    

## 9. How2Eval 
### a. Evaluation on the VOC07+12 test set
1. This article uses the VOC format for evaluation. VOC07+12 has divided the test set, there is no need to use `voc_annotation.py` to generate the txt in the ImageSets folder.
2. Modify `model_path` and `classes_path` in `yolo.py`. `model_path` points to the trained weight file, in the logs folder. The `classes_path` points to the txt corresponding to the detection category.    
3. Run `get_map.py` to get the evaluation result, which will be saved in the `map_out` folder.    

### b. Evaluation on custom dataset
1. This article uses the VOC format for evaluation      
2. If the voc_annotation.py file has been run before training, the code will automatically divide the data set into training set, validation set and test set. If you want to modify the ratio of the test set, you can modify trainval_percent in the voc_annotation.py file. trainval_percent is used to specify the ratio of (training set + validation set) to test set, by default (training set + validation set): test set = 9:1. train_percent is used to specify the ratio of training set to validation set in (training set + validation set). By default, training set: validation set = 9:1.
3. After dividing the test set with voc_annotation.py, go to the get_map.py file to modify the classes_path. The classes_path is used to point to the txt corresponding to the detection category. This txt is the same as the txt during training. Evaluate your own data set must be modified.
4. Modify model_path and classes_path in yolo.py. model_path points to the trained weight file, in the logs folder. The classes_path points to the txt corresponding to the detection category.  
5. Run get_map.py to get the evaluation result, which will be saved in the map_out folder

## 10. Reference
https://github.com/qqwweee/keras-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch
