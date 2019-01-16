#ifndef COSINE_H
#define COSINE_H

#include <cmath>
#include <vector>
#include <iostream>
#include <map>

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>       /* time_t, time, ctime */ 

#include "mtcnn.h"
#include "feature.h"

#define PI 3.14159265f



// 夹角余弦
double cosine(float v1[], float v2[]);

//人脸模糊度判断
int blurDetect(cv::Mat srcImage);

//人脸侧脸角度模糊质量筛选
int screen(vector<FaceDetector::BoundingBox> res, cv::Mat img, cv::Mat &roi, int k);

int blurdectect(cv::Mat &img);

#endif
