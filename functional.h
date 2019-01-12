#ifndef COSINE_H
#define COSINE_H

#include <cmath>
#include <vector>
#include <stdio.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>       /* time_t, time, ctime */ 

#include "mtcnn.h"

#define PI 3.14159265f


// 夹角余弦
double cosine(float v1[], float v2[]);

//人脸特征二进制文件比对
int contrast(std::vector<float> face_feature, cv::Mat *roi);

//人脸模糊度判断
int blurDetect(cv::Mat srcImage);

//人脸侧脸角度模糊质量筛选
int screen(vector<FaceDetector::BoundingBox> res, cv::Mat img, cv::Mat &roi, int k);

int blurdectect(cv::Mat &img);

#endif
