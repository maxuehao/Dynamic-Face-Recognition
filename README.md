# Dynamic-Face-Recognition
Deep learning,Face recognition algorithm based on Mobilenet and Mtcnn(LFW 95%, Mobilenet-based face recognition model is superior to the accuracy of the original centerloss paper)

# Rely
1.BVLC Caffe c++ api  
2.Opencv2.4 

# Base
1.Face detection: Mtcnn  
2.Face recognition: centerloss model base on mobilenet 

# Performance
1.Face recognition: 99.5% LFW  
2.Face recognition forward speed: Movidius NCS1 45ms/img  
3.Nvidia-tk1: Detection and recognition 80ms/frame  

此版本为早期代码，近期将维护更新...
