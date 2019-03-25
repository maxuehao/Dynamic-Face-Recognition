# Dynamic-Face-Recognition
Deep learning,Face recognition algorithm based on Mobilenet and Mtcnn(LFW 99.1%, Mobilenet-based face recognition model is superior to the accuracy of the original centerloss paper)

## Centorloss&Mtcnn Website
https://github.com/ydwen/caffe-face<br>
https://github.com/dlunion/mtcnn<br>

## Environmental requirements
  1.Caffe(Be based on CUDA&Cudnn)<br>2.OpenCV 3.x<br>3.Cmake<br>

## Caffe environment variable setting
  1.Add 
  ```
    export PYTHONPATH=/home/pv/ma/caffe-master/python:$PYTHONPATH
    export CAFFE_ROOT=/home/pv/ma/caffe-master
  ```
   to ~/.bashrc
  
## Instructions for use
1.Please clip and align the face images that need to be compared and put them in the face_id folder  
```
   mkdir build
   cmake ..
   ./DFR
```
## DEMO
![image](https://github.com/maxuehao/Dynamic-Face-Recognition/blob/master/demo.png)
