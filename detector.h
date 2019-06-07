#ifndef __DETECTOR_HPP__
#define __DETECTOR_HPP__


#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "detector_interface.h"

/* 自定义结构 */
//人脸的5个关键点坐标
typedef struct tagFaceLandmark
{
    float x[5];
    float y[5];
}FaceLandmark;

typedef struct tagScaleWindow
{
    int h;
    int w;
    float scale;
}ScaleWindow;

typedef struct tagFaceBox
{
    float x0;                           //检测框左上角x0坐标
    float y0;                           //检测框左上角y0坐标
    float x1;                           //检测框右下角x1坐标
    float y1;                           //检测框右下角y1坐标
    float score;                        //检测框的置信度
    float regress[4];
    float px0;
    float py0;
    float px1;
    float py1;
    FaceLandmark landmark;              //人脸的5个关键点
}FaceBox;

class detector : public DetectorPluginInterface {
	public:
		detector(void){}

		int load_model(const std::string& model_path) override;
		int detect(cv::Mat& img, std::vector<face_box>& face_list) override;
		void release() override;
		~detector(void){}
		
	protected:
        void CopyOnePatch(const cv::Mat &img,FaceBox&input_box,float * data_to, int width, int height);
        int  RunPNet(const cv::Mat &img, ScaleWindow &win, std::vector<FaceBox> &boxList);
        void RunRNet(const cv::Mat &img,std::vector<FaceBox> &pnet_boxes, std::vector<FaceBox> &output_boxes);
        void RunONet(const cv::Mat &img,std::vector<FaceBox> &rnet_boxes, std::vector<FaceBox> &output_boxes);

	private:
	    int m_minSize;
        float m_factor;
        float m_pnetThreshold;
        float m_rnetThreshold;
        float m_onetThreshold;
		graph_t PNet_graph;
		graph_t RNet_graph;
		graph_t ONet_graph;
};

extern "C" int InitDetexctor(DetectorPluginInterface* Detector);

#endif