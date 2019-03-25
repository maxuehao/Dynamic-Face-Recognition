#include "functional.h"

using namespace std;

int main() {

    //加载mtcnn检测模型
    FaceDetector fd("../model", FaceDetector::MODEL_V1);
    //加载人脸识别特征提取模型
    Classifier classifier("../model/deploy.prototxt", "../model/face.caffemodel");

    //计算face id文件夹下人脸图片的特征并存入map结构里
    //存放人脸对比图片的文件夹
    char *dir_name= "../face_id/";
    map<string, vector<float>> face_id;
    // check the parameter !
	if( NULL == dir_name )
	{
		cout<<" dir_name is null ! "<<endl;
	}
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat( dir_name , &s );
	if( ! S_ISDIR( s.st_mode ) )
	{
		cout<<"dir_name is not a valid directory !"<<endl;
	}
	
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir( dir_name );
	if( NULL == dir )
	{
		cout<<"Can not open dir "<<dir_name<<endl;
	}
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL )
	{
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
        char *name = (char *) malloc(100 * sizeof(char));
        strcpy(name, dir_name);
        strcat(name, filename ->d_name);
        printf("%s\n", name);
        cv::Mat img;
        img = cv::imread(name);
        string face_name = filename ->d_name;
        vector<float> predictions = classifier.Classify(img);
        face_id.insert(pair<string, vector<float>>(face_name, predictions));    
        free(name);
	}

    //定义CV摄像头参数
    int frame_num = 0;
    cv::VideoCapture cap(0); 
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);  
    cv::Mat frameImg;

    //人脸比对相似度阈值
    float face_confidence = 0.8;

    while(true)
    {
        cap >> frameImg;
        cv::Mat testImg = frameImg;
        cv::Mat img=testImg.clone();
        //调用Mtcnn人脸检测端口,70为所检测人脸最小尺寸
        vector<FaceDetector::BoundingBox> res = fd.Detect(testImg, FaceDetector::BGR, FaceDetector::ORIENT_UP ,70, 0.8, 0.9, 0.95);
        cout<< "FPS NUM:" << frame_num << endl;
        cout<< "Detected face NUM : " << res.size() << endl;

        if (res.size()!= 0)
        {   
            for(int k = 0; k < res.size(); k++)
            {
                //绘制人脸检测框
                cv::rectangle(testImg, cv::Point(res[k].x1, res[k].y1), cv::Point(res[k].x2, res[k].y2), cv::Scalar(255, 255, 0), 3);
                //绘制人脸关键点
                for (int i = 0; i < 5; i++)
                {
                    cv::circle(testImg, cv::Point(res[k].points_x[i], res[k].points_y[i]), 1, cv::Scalar(0, 0, 255), 2);
                }
                cv::Mat roi;
                //人脸质量筛选
                int signal = screen(res, img, roi, k);
                if (signal == 0) continue;
                //cv::imwrite("1.jpg",roi);
                //输出512维度人脸特征向量
                vector<float> predictions = classifier.Classify(roi);
                map<string, vector<float>>::iterator iter; 
                string name; 
                float score;
                for(iter = face_id.begin(); iter != face_id.end(); iter++){
                    float feature_1[512], feature_2[512];
                    for(int i=0; i< 512;i++)  
                    {  
                        feature_1[i] = predictions[i]; 
                        feature_2[i] = iter->second[i];
                    } 
                    float cosin = cosine(feature_1, feature_2);
                    if(cosin >= face_confidence){
                        name = iter->first;
                        score = cosin;
                        printf("cosin:%f\n",cosin);
                        break;
                    }
                    else{
                        score = 0;
                        name = "None";
                    }


                }  
                //人脸ID打印
                string text = "ID:" + name;
                cv::Point p = cv::Point(res[k].x1, res[k].y1-5);
                cv::putText(testImg, text, p, cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(0, 255, 0), 2, CV_AA);
                //置信度打印
                string str_score =to_string(score);
                string score_text = "Score:" + str_score;
                cv::Point z = cv::Point(res[k].x1, res[k].y1-25);
                cv::putText(testImg, score_text, z, cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(0, 0, 255), 2, CV_AA);
            }
        }
        cv::imshow("test", testImg);
        frame_num++;
        cv::waitKey(1);
    }
    return 0;
}
