#ifndef __DETECTOR_PLIUGIN_INTERFACE_HPP__
#define __DETECTOR_PLIUGIN_INTERFACE_HPP__

#include <string>
#include <vector>

#include <unistd.h>
#include <functional>
#include <algorithm>
#include <fstream>

#include "tengine_c_api.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct face_box
{
	float x0;
	float y0;
	float x1;
	float y1;

	/* confidence score */
	float score;
};

class DetectorPluginInterface {
	public:
		DetectorPluginInterface(void){}
		~DetectorPluginInterface(void){}

		virtual int load_model(const std::string& model_file) = 0;
		virtual int detect(cv::Mat& img, std::vector<face_box>& face_list) = 0;
		virtual void release() = 0;
	
	    DetectorPluginInterface* me;
};


#endif //__DETECTOR_PLIUGIN_INTERFACE_HPP__