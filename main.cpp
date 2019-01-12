#include "mtcnn.h"
#include "feature.h"
#include "functional.h"
 




int main(int argc, char **argv) {
    //加载mtcnn检测模型
    FaceDetector fd("model", FaceDetector::MODEL_V1);
    //加载人脸识别特征提取模型
    string model_file   = "model/deploy.prototxt";
    string trained_file = "model/face.caffemodel";
    Classifier classifier(model_file, trained_file);

    int frame_num = 0;
    //判断缓冲区是否写入信息
    int temp = 0;
    int id = 0;
    //创建缓冲数组储存前一帧检测框信息
    float buf[100][5];
    //记录缓冲区长度
    int buf_num;

    cv::VideoCapture cap(0); 
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);  
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);  
    cv::Mat frameImg;

    while(true)
    {
        cap >> frameImg;
        cv::Mat testImg = frameImg(cv::Rect(30,260,1240,400));
        cv::Mat img=testImg.clone();
        //cout << "h" << testImg.rows << endl;
        //cout << "w" << testImg.cols << endl;
        vector<FaceDetector::BoundingBox> res = fd.Detect(testImg, FaceDetector::BGR, FaceDetector::ORIENT_UP ,140, 0.8, 0.9, 0.95);
        cout<< "FPS NUM:" << frame_num << endl;
        cout<< "Detected face NUM : " << res.size() << endl;
        if (res.size()!= 0)
        {   
            //初始化缓冲区
            if (temp ==0)
            {
                for(int k = 0; k < res.size(); k++)
                {
                    buf[k][0] = res[k].x1;
                    buf[k][1] = res[k].y1;
                    buf[k][2] = res[k].x2;
                    buf[k][3] = res[k].y2;
                    buf[k][4] = id;
                    id++;
                } 
                cout<<"################ First uodat！！！#################"<<endl;
                //cout<<buf[0][0]<<endl;
                buf_num = res.size();
                temp = 1;
                frame_num++;
                continue;
            }
            //cout<<buf_num<<endl;
            //创建更新缓冲区所用的交换数组
            float swap_buf [res.size()][5];
            for(int k = 0; k < res.size(); k++)
            {
                int id_name;
                int iou_sig = 0;

                int bbox1[4] = {(int)res[k].x1,(int)res[k].y1, (int)res[k].x2, (int)res[k].y2};
                //计算检测框与缓冲区框的IOU
                for(int j =0; j < buf_num; j++)
                {
                    int bbox2[4] = {(int)buf[j][0], (int)buf[j][1], (int)buf[j][2], (int)buf[j][3]};
                    float con = iou(bbox1, bbox2);
                    //cout<<"IOU:"<<con<<endl;
                    if(con>0.1)
                    {           
                        //绑定相同ID             
                        id_name = buf[j][4];
                        iou_sig = 1;
                    }  
                }
                if (iou_sig == 0)
                {
                    id_name = id++;
                }
                //写入交换数组
                swap_buf[k][0] = res[k].x1;
                swap_buf[k][1] = res[k].y1;
                swap_buf[k][2] = res[k].x2;
                swap_buf[k][3] = res[k].y2;
                swap_buf[k][4] = id_name;

                //std::string text = "ID:" + std::to_string(id_name);
                //cv::Point p = cv::Point(res[k].x1, res[k].y1);
                //cv::putText(testImg, text, p, cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(0, 255, 0), 2, CV_AA);

                cv::rectangle(testImg, cv::Point(res[k].x1, res[k].y1), cv::Point(res[k].x2, res[k].y2), cv::Scalar(255, 0, 0), 2);
                for (int i = 0; i < 5; i++)
                {
                    cv::circle(testImg, cv::Point(res[k].points_x[i], res[k].points_y[i]), 1, cv::Scalar(0, 0, 255), 2);
                }
                cv::Mat roi;
                int signal = screen(res, img, roi, k);
                if (signal == 0) continue;
                std::vector<float> predictions = classifier.Classify(roi);
                //人脸识别比对
                int name = contrast(predictions, &roi);
                //std::cout <<name<< std::endl;
                
                //人脸ID打印
                std::string text = "ID:" + std::to_string(name);
                cv::Point p = cv::Point(res[k].x1, res[k].y1-5);
                cv::putText(testImg, text, p, cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(0, 255, 0), 2, CV_AA);

            }
            //将交换数组信息写入缓冲区更新
            for(int k = 0; k < res.size(); k++)
            {
                buf[k][0] = swap_buf[k][0];
                buf[k][1] = swap_buf[k][1];
                buf[k][2] = swap_buf[k][2];
                buf[k][3] = swap_buf[k][3];
                buf[k][4] = swap_buf[k][4];
            }
            buf_num = res.size();
        }
        cv::imshow("test", testImg);
        frame_num++;
        cv::waitKey(1);
    }
    return 0;
}
