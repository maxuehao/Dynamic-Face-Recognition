#include "functional.h"

// 夹角余弦
double cosine(float v1[], float v2[])
{
  double product = 0;
  double v1_sum = 0;
  double v2_sum = 0;

  for (int i = 0; i < 512; ++i) {

    double temp = v1[i]*v2[i];
    double v1_temp = v1[i]*v1[i];
    double v2_temp = v2[i]*v2[i];

    product += temp;
    v1_sum += v1_temp;
    v2_sum += v2_temp;
  }

  double cosine = product/sqrt(v1_sum *v2_sum);
  return cosine;
}

//二进制人脸识别数据操作
int contrast(std::vector<float> face_feature, cv::Mat *roi)  
{  
    FILE *p = NULL;  
    int name = 0;
    p = fopen("feature.bat","a+b");
    float read_feature[512];
    float feature[512];

    for(int i=0; i< 512;i++)  
    {  
        feature[i] = face_feature[i]; //初始化缓存区  
    }  
  
    int temp = 0;
	while(!feof(p))
	{   
        if (fread(&read_feature,512*sizeof(float),1,p) == 0) break;
		    float cosin = cosine(feature, read_feature);
        //std::cout<<cosin<<std::endl;
		if (cosin > 0.58)
        {
            temp = 1;
            break;
        }
        name++;	
	}
    if (temp == 0)
    {
        fwrite(&feature,512*sizeof(float), 1, p);
        std::cout<<"---------------"<<std::endl;
        time_t rawtime;
        time(&rawtime);  
        cv::imwrite(std::to_string(rawtime)+".jpg",*roi);
    }
    fclose (p);
    return name;  
}  

//人脸模糊拉普拉斯判断
int blurDetect(cv::Mat &srcImage)  
{  
  
    cv::Mat gray1;  
    if (srcImage.channels() != 1)  
    {  
        cvtColor(srcImage, gray1, CV_RGB2GRAY);  
    }  
    else  
    {  
        gray1 = srcImage.clone();  
    }  
    cv::Mat tmp_m1, tmp_sd1;   
    double m1 = 0, sd1 = 0;   
    cv::Laplacian(gray1, gray1, CV_16S, 3);  
    cv::convertScaleAbs(gray1, gray1);  
    cv::meanStdDev(gray1, tmp_m1, tmp_sd1);  
    m1 = tmp_m1.at<double>(0, 0); 
    sd1 = tmp_sd1.at<double>(0, 0);    
    if (sd1*sd1 < 1900)  
    {  
        return 0;  
    }  
    else  
    {  
        return 1;   
    }  
} 

//人脸侧脸角度模糊质量筛选
int screen(vector<FaceDetector::BoundingBox> res, cv::Mat img, cv::Mat &roi, int k)
{

    int temp = 0;
    //人脸角度矫正 
    //两眼之间距离
    float eye_distance = sqrt(pow((res[k].points_x[1]-res[k].points_x[0]), 2) + pow((res[k].points_y[1]-res[k].points_y[0]),2));
    //std::cout << eye_distance << std::endl;
    //计算旋转弧度
    float rotation = atan2((res[k].points_y[1]-res[k].points_y[0]),(res[k].points_x[1]-res[k].points_x[0]));
    float angle = (rotation/PI)*180;
    //std::cout << angle << std::endl;
    //旋转图像
    cv::Mat newIm;
    cv::Point2f pt(res[k].points_x[0], res[k].points_y[0]);
    cv::Mat r = cv::getRotationMatrix2D(pt,angle,1.0);
    cv::warpAffine(img,newIm,r,cv::Size(img.cols,img.rows));
    //取人脸识别roi
    //判断人脸roi坐标是否越界
    if ((res[k].points_x[0]-(0.7*eye_distance))<0 || (res[k].points_x[0]+(1.7*eye_distance))>img.cols || (res[k].points_y[0]-(0.7*eye_distance))<0 || (res[k].points_y[0]+(2*eye_distance))>img.rows) 
    {
        temp = 0;
    }
    else
    {    //筛选侧脸
        if (res[k].points_x[0]>res[k].points_x[2] || res[k].points_x[3] > res[k].points_x[2] || res[k].points_x[2] > res[k].points_x[1] || res[k].points_x[2] >res[k].points_x[4] )
        {
             temp = 0;
        }
        else
        {    
            if ( -0.27*(res[k].points_x[1]-res[k].points_x[0]) > (res[k].points_x[2]-res[k].points_x[0])- (res[k].points_x[1]-res[k].points_x[2]) || (res[k].points_x[2]-res[k].points_x[0])- (res[k].points_x[1]-res[k].points_x[2]) > 0.27*(res[k].points_x[1]-res[k].points_x[0]) )
            {
                 temp = 0;  
            }    
            else
            {
                //std::cout<<(res[k].points_x[2]-res[k].points_x[0])- (res[k].points_x[1]-res[k].points_x[2])<<std::endl;
                roi = newIm(cv::Rect((int)(res[k].points_x[0]-(0.7*eye_distance)), (int)(res[k].points_y[0]-(0.7*eye_distance)), (int)(2.4*eye_distance), (int)(2.7*eye_distance)));
                //人脸模糊判断
                if (blurdectect(roi) == 0)
                {
                    std::cout<<"####模糊模糊####"<<std::endl;
                    temp = 0;
                }
                else
                {
                    temp = 1;
                }
            }
        }
    }
    return temp;
}

//sobel算子
int blurdectect( cv::Mat &img)
{    
    cv::Mat  halfImg=img(cv::Rect(0,0,img.cols,img.rows/2)).clone();
    cv::Mat imageGrey;  
    cv::cvtColor(halfImg, imageGrey, CV_RGB2GRAY);  
    cv::Mat meanValueImage;  
    cv::Mat meanStdValueImage;  
    cv::Mat imageSobel;  
    cv::Sobel(imageGrey, imageSobel, CV_8U, 1, 1);   
    cv::meanStdDev(imageSobel, meanValueImage, meanStdValueImage);  
    double meanValue = 0.0;  
    meanValue = meanValueImage.at<double>(0, 0);  
    if(meanValue<4.0)
        return 0;
    else
        return 1;
       
}