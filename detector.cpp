#include "detector.h"
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string.h>
#define TEST
using namespace std;
void showbox(cv::Mat& src,std::vector<FaceBox>& face_list)
{
    for(int i = 0; i < ( int )face_list.size(); i++)
    {
        
        FaceBox box = face_list[i];
        
        cv::rectangle(src, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
                    2);

        std::ostringstream score_str;
        score_str.precision(3);
        score_str << box.score;
        std::string label = score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
        cv::rectangle(src,
                    cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
                            cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(src, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
        // printf("Face %d:\t%.0f%%\t", i, box.score * 100);
        // printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
    }
    // det_faces.clear();
    // cv::imwrite(image_save[picnum],img);
    cv::imshow("ret",src);
    cv::waitKey(0);
}
/*获取文件名*/
void GetFileNames(string path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

/*获取保存图片名*/
void GetSaveNames(std::string path_read, std::string path_save, std::vector<string>& savenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path_read.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            savenames.push_back(path_save + "/" + ptr->d_name);
    }
    closedir(pDir);
}

/* 此函数必须保留  */
extern "C" int InitDetector(DetectorPluginInterface** Detector)
{
    *Detector = new detector();
    return 0; //ok
}

#define NMS_UNION 1
#define NMS_MIN  2

void nms_boxes(std::vector<FaceBox>& input, float threshold, int type, std::vector<FaceBox>&output);
void regress_boxes(std::vector<FaceBox>& rects);
void square_boxes(std::vector<FaceBox>& rects);
void padding(int img_h, int img_w, std::vector<FaceBox>& rects);
void process_boxes(std::vector<FaceBox>& input, int img_h, int img_w, std::vector<FaceBox>& rects);
void generate_bounding_box(const float * confidence_data,
        const float * reg_data, float scale, float threshold,
        int feature_h, int feature_w, std::vector<FaceBox> &output, bool transposed);

void set_input_buffer(std::vector<cv::Mat>& input_channels,
		float* input_data, const int height, const int width);
void  cal_pyramid_list(int height, int width, int min_size, float factor,std::vector<ScaleWindow>& list);


/* 必须实现，载入参赛者模型，如有多个模型请添加多个graph到头文件中，并在此函数中分别create_graph。  */
int detector::load_model(const std::string& model_path)
{
    //PNet
    std::string model_file = model_path + "det1.tmfile";
    PNet_graph = create_graph(nullptr, "tengine", model_file.c_str());
    if(PNet_graph == nullptr)
    {
        std::cout << "Create PNet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    //RNet
    model_file = model_path + "det2.tmfile";
    RNet_graph = create_graph(nullptr, "tengine", model_file.c_str());
    if(RNet_graph == nullptr)
    {
        std::cout << "Create RNet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    //ONet
    model_file = model_path + "det3.tmfile";
    ONet_graph = create_graph(nullptr, "tengine", model_file.c_str());
    if(ONet_graph == nullptr)
    {
        std::cout << "Create ONet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    m_minSize = 20;
    m_factor = 0.709;
    // m_pnetThreshold = 0.5;
    // m_rnetThreshold = 0.3;
    // m_onetThreshold = 0.3;
    m_pnetThreshold = 0.6;
    m_rnetThreshold = 0.6;
    m_onetThreshold = 0.5;

    return 0;
}

/* 必须实现，对传入的图像，将检测结果保存在face_list中，face_box的定义在detector_inferface.h中。  
    这里调用了两个函数get_input_data和post_process_ssd，可根据模型自由修改。
*/
int detector::detect(cv::Mat& src,std::vector<face_box>& face_list)
{
    float mean = 127.5;
    float alpha = 0.0078125;
    cv::Mat working_img;
    std::vector<ScaleWindow> win_list;
    std::vector<FaceBox> total_pnet_boxes;
    std::vector<FaceBox> total_rnet_boxes;
    std::vector<FaceBox> total_onet_boxes;
    std::vector<FaceBox> rnet_boxes;

    std::vector<FaceBox> final_face_list;

    src.convertTo(working_img, CV_32FC3);
    working_img=(working_img-mean)*alpha;
    //mtcnn_change
    // working_img=working_img.t();
    
    if(!working_img.empty())
    {
        if (4 == working_img.channels())
        {
            cv::cvtColor(working_img, working_img, cv::COLOR_BGRA2RGB);
        }
        else if (1 == working_img.channels())
        {
            cv::cvtColor(working_img, working_img, cv::COLOR_GRAY2RGB);
        }
        else
        {
            cv::cvtColor(working_img, working_img, cv::COLOR_BGR2RGB);
        }
    }
    
    //
    int img_h=working_img.rows;
    int img_w=working_img.cols;

    //run PNet
    cal_pyramid_list(img_h, img_w, m_minSize, m_factor, win_list);
    // std::cout << "cal_pyramid_list m_minSize: " << m_minSize  << "factor: " << m_factor << std::endl;
    for(size_t i=0;i<win_list.size();i++)
    {
        std::vector<FaceBox>boxes;
        RunPNet(working_img,win_list[i],boxes);
        // std::cout << "RunPNet ret: " << ret << std::endl;
        total_pnet_boxes.insert(total_pnet_boxes.end(),boxes.begin(),boxes.end());
    }
    
    std::vector<FaceBox> pnet_boxes;
    process_boxes(total_pnet_boxes,img_h,img_w,pnet_boxes);
    if(!pnet_boxes.size())
    {
        // std::cout << "pnet output None" << std::endl;
        return 1;
    }
    

    //run RNet
    RunRNet(working_img,pnet_boxes,total_rnet_boxes);
    process_boxes(total_rnet_boxes,img_h,img_w,rnet_boxes);
    if(!rnet_boxes.size())
    {
        return 2;
    }
    // showbox(src,rnet_boxes);

    //run ONet
    RunONet(working_img,rnet_boxes,total_onet_boxes);
    if(!total_onet_boxes.size())
    {
        return 3;
    }

    //calculate the landmark
    for(unsigned int i=0;i<total_onet_boxes.size();i++)
    {
        FaceBox& box=total_onet_boxes[i];
        float h=box.x1-box.x0+1;
        float w=box.y1-box.y0+1;

        for(int j=0;j<5;j++)
        {
            box.landmark.x[j]=box.x0+w*box.landmark.x[j]-1;
            box.landmark.y[j]=box.y0+h*box.landmark.y[j]-1;
        }
    }

    //Get Final Result
    regress_boxes(total_onet_boxes);
    nms_boxes(total_onet_boxes, 0.5, NMS_MIN, final_face_list);

    // showbox(src,final_face_list);
    //nms_boxes(total_onet_boxes, 0.7, NMS_MIN, final_face_list);
    //mtcnn_change
    for(unsigned int i=0;i<final_face_list.size();i++)
    {
        FaceBox& box=final_face_list[i];
        // std::swap(box.x0,box.y0);
        // std::swap(box.x1,box.y1);
        for(int l=0;l<5;l++)
        {
            std::swap(box.landmark.x[l],box.landmark.y[l]);
        }

        face_box bbbox;
        bbbox.x0 = box.x0;
        bbbox.y0 = box.y0;
        bbbox.x1 = box.x1;
        bbbox.y1 = box.y1;
        bbbox.score = box.score;
        face_list.push_back(bbbox);
    }
}

/* 必须实现，释放资源，如只有一个模型可不修改。  */
void detector::release()
{
    int ret = postrun_graph(PNet_graph);
    if(ret != 0)
    {
        std::cout << "Postrun PNet_graph failed, errno: " << get_tengine_errno() << "\n";
    }
    destroy_graph(PNet_graph);

    ret = postrun_graph(RNet_graph);
    if(ret != 0)
    {
        std::cout << "Postrun RNet_graph failed, errno: " << get_tengine_errno() << "\n";
    }
    destroy_graph(RNet_graph);

    ret = postrun_graph(ONet_graph);
    if(ret != 0)
    {
        std::cout << "Postrun ONet_graph failed, errno: " << get_tengine_errno() << "\n";
    }
    destroy_graph(ONet_graph);

    release_tengine();
}

void set_cvMat_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width)
{
    for(int i = 0; i < 3; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
}

int detector::RunPNet(const cv::Mat &img, ScaleWindow &win, std::vector<FaceBox> &boxList)
{
    cv::Mat  resized;
    int scale_h=win.h;
    int scale_w=win.w;
    float scale=win.scale;
	static bool first_run = true;
    cv::resize(img, resized, cv::Size(scale_w, scale_h), 0, 0);

    /* input */
    //tensor_t input_tensor=get_graph_tensor(PNet_graph,"data");
    tensor_t input_tensor=get_graph_input_tensor(PNet_graph, 0, 0);
    int dims[]={1,3,scale_h,scale_w};
    set_tensor_shape(input_tensor,dims,4);
    int in_mem=sizeof(float)*scale_h*scale_w*3;
    float* input_data=(float*)malloc(in_mem);

    std::vector<cv::Mat> input_channels;
    set_cvMat_input_buffer(input_channels, input_data, scale_h, scale_w);
    //set_input_buffer(input_channels, input_data, scale_h, scale_w);
    cv::split(resized, input_channels);

    set_tensor_buffer(input_tensor,input_data,in_mem);
    //prerun_graph(PNet_graph);
    if(first_run)
    {
        if(prerun_graph(PNet_graph) != 0)
        {
            std::cout << "Prerun PNet graph failed, errno: " << get_tengine_errno() << "\n";
            return -1;
        }
        first_run = false;
    }
    if(run_graph(PNet_graph, 1) != 0)
    {
        std::cout << "Run PNet graph failed, errno: " << get_tengine_errno() << "\n";
        return -2;
    }
    free(input_data);
    put_graph_tensor(input_tensor);

    /* output */
    //tensor_t tensor=get_graph_tensor(PNet_graph,"conv4-2");
    tensor_t tensor=get_graph_tensor(PNet_graph, "tensor_23");
    get_tensor_shape(tensor,dims,4);
    float *  reg_data=(float *)get_tensor_buffer(tensor);
    int feature_h=dims[2];
    int feature_w=dims[3];
    //put_graph_tensor(tensor);

    //tensor=get_graph_tensor(PNet_graph,"prob1");
    tensor=get_graph_tensor(PNet_graph, "tensor_22");

    float *  prob_data=(float *)get_tensor_buffer(tensor);
    std::vector<FaceBox> candidate_boxes;
    generate_bounding_box(prob_data, reg_data, scale, m_pnetThreshold, feature_h,feature_w,candidate_boxes,false);

    //put_graph_tensor(tensor);
    //postrun_graph(PNet_graph);
    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);
    nms_boxes(candidate_boxes, 0.35, NMS_UNION, boxList);
    return 0;
}

void detector::CopyOnePatch(const cv::Mat& img,FaceBox&input_box,float * data_to, int width, int height)
{
    std::vector<cv::Mat> channels;

    set_input_buffer(channels, data_to, height, width);

    cv::Mat chop_img = img(cv::Range(input_box.py0,input_box.py1),
            cv::Range(input_box.px0, input_box.px1));

    int pad_top = std::abs(input_box.py0 - input_box.y0);
    int pad_left = std::abs(input_box.px0 - input_box.x0);
    int pad_bottom = std::abs(input_box.py1 - input_box.y1);
    int pad_right = std::abs(input_box.px1-input_box.x1);

    cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom,pad_left, pad_right,  cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::resize(chop_img, chop_img, cv::Size(width, height), 0, 0);
    cv::split(chop_img, channels);
}


void detector::RunRNet(const cv::Mat& img, std::vector<FaceBox>& pnet_boxes, std::vector<FaceBox>& output_boxes)
{
    int batch=pnet_boxes.size();
    int channel = 3;
    int height = 24;
    int width = 24;
    int conf_page_size=2;
    int reg_page_size=4;
	static bool first_run = true;
    
    //tensor_t input_tensor = get_graph_tensor(RNet_graph,"data");
    tensor_t input_tensor=get_graph_input_tensor(RNet_graph, 0, 0);
    int dims[] = {batch,channel,height,width};
    int img_size = channel*height*width;
    int in_mem = sizeof(float)*batch*img_size;
    float* input_data = (float*)malloc(in_mem);
    float* input_ptr = input_data;

    set_tensor_shape(input_tensor,dims,4);
    set_tensor_buffer(input_tensor,input_ptr,in_mem);

    for(int i=0;i<batch;i++)
    {
        CopyOnePatch(img,pnet_boxes[i],input_ptr,height,width);
        input_ptr += img_size;
    }

    //prerun_graph(RNet_graph);
    if(first_run)
    {
        if(prerun_graph(RNet_graph) != 0)
        {
            std::cout << "Prerun RNet graph failed, errno: " << get_tengine_errno() << "\n";
            return;
        }
        first_run = false;
    }
    if(run_graph(RNet_graph, 1) != 0)
    {
        std::cout << "Run RNet graph failed, errno: " << get_tengine_errno() << "\n";
        return;
    }
    free(input_data);
    //put_graph_tensor(input_tensor);

    /* output */
    //tensor_t tensor=get_graph_tensor(RNet_graph,"conv5-2");
    tensor_t tensor=get_graph_tensor(RNet_graph, "tensor_29");
    float *reg_data=(float *)get_tensor_buffer(tensor);
    //put_graph_tensor(tensor);

    //tensor=get_graph_tensor(RNet_graph,"prob1");
    tensor=get_graph_tensor(RNet_graph, "tensor_28");
    float *confidence_data=(float *)get_tensor_buffer(tensor);
    //put_graph_tensor(tensor);

    for(int i=0;i<batch;i++)
    {
        if (*(confidence_data+1) > m_rnetThreshold)
        {
            FaceBox output_box;
            FaceBox& input_box=pnet_boxes[i];
            output_box.x0=input_box.x0;
            output_box.y0=input_box.y0;
            output_box.x1=input_box.x1;
            output_box.y1=input_box.y1;
            output_box.score = *(confidence_data+1);
            /*Note: regress's value is swaped here!!!*/
            // output_box.regress[0]=reg_data[1];
            // output_box.regress[1]=reg_data[0];
            // output_box.regress[2]=reg_data[3];
            // output_box.regress[3]=reg_data[2];
            output_box.regress[0]=reg_data[0];
            output_box.regress[1]=reg_data[1];
            output_box.regress[2]=reg_data[2];
            output_box.regress[3]=reg_data[3];
            output_boxes.push_back(output_box);
        }

        confidence_data += conf_page_size;
        reg_data += reg_page_size;
    }

    //postrun_graph(RNet_graph);
	release_graph_tensor(input_tensor);
	release_graph_tensor(tensor);
}


void detector::RunONet(const cv::Mat& img, std::vector<FaceBox>& rnet_boxes, std::vector<FaceBox>& output_boxes)
{
    int batch=rnet_boxes.size();
    int channel = 3;
    int height = 48;
    int width = 48;
    static bool first_run = true;
    //dump_graph(ONet_graph);
    
    //tensor_t input_tensor = get_graph_tensor(ONet_graph,"data");
    tensor_t input_tensor=get_graph_input_tensor(ONet_graph, 0, 0);
    int dims[] = {batch,channel,height,width};
    int img_size = channel*height*width;
    int in_mem = sizeof(float)*batch*img_size;
    float* input_data = (float*)malloc(in_mem);
    float* input_ptr = input_data;

    set_tensor_shape(input_tensor,dims,4);
    set_tensor_buffer(input_tensor,input_ptr,in_mem);

    for(int i=0;i<batch;i++)
    {
        CopyOnePatch(img,rnet_boxes[i],input_ptr,height,width);
        input_ptr += img_size;
    }

    //prerun_graph(ONet_graph);
     if(first_run)
    {
        if(prerun_graph(ONet_graph) != 0)
        {
            std::cout << "Prerun ONet graph failed, errno: " << get_tengine_errno() << "\n";
            return;
        }
        first_run = false;
    }

    if(run_graph(ONet_graph, 1) != 0)
    {
        std::cout << "Run ONet graph failed, errno: " << get_tengine_errno() << "\n";
        return;
    }
    free(input_data);

    /* output */
    //tensor_t tensor=get_graph_tensor(ONet_graph,"conv6-3");
    tensor_t tensor=get_graph_tensor(ONet_graph, "tensor_38");
    float *  points_data=(float *)get_tensor_buffer(tensor);

    //tensor=get_graph_tensor(ONet_graph,"prob1");
    tensor=get_graph_tensor(ONet_graph, "tensor_37");
    float *  confidence_data=(float *)get_tensor_buffer(tensor);

    //tensor=get_graph_tensor(ONet_graph,"conv6-2");
    tensor=get_graph_tensor(ONet_graph, "tensor_39");
    float *  reg_data=(float *)get_tensor_buffer(tensor);

    int conf_page_size = 2;
    int reg_page_size = 4;
    int points_page_size = 10;

    for(int i=0;i<batch;i++)
    {
        if (*(confidence_data+1) > m_onetThreshold)
        {
            FaceBox output_box;
            FaceBox& input_box = rnet_boxes[i];

            output_box.x0 = input_box.x0;
            output_box.y0 = input_box.y0;
            output_box.x1 = input_box.x1;
            output_box.y1 = input_box.y1;

            output_box.score =* (confidence_data+1);

            // output_box.regress[0] = reg_data[1];
            // output_box.regress[1] = reg_data[0];
            // output_box.regress[2] = reg_data[3];
            // output_box.regress[3] = reg_data[2];

            output_box.regress[0] = reg_data[0];
            output_box.regress[1] = reg_data[1];
            output_box.regress[2] = reg_data[2];
            output_box.regress[3] = reg_data[3];

            /*Note: switched x,y points value too..*/
            for (int j = 0; j<5; j++)
            {
                output_box.landmark.x[j] = *(points_data + j+5);
                output_box.landmark.y[j] = *(points_data + j);
            }

            output_boxes.push_back(output_box);
        }

        confidence_data +=conf_page_size;
        reg_data += reg_page_size;
        points_data += points_page_size;
    }

    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);
}

void nms_boxes(std::vector<FaceBox>& input, float threshold, int type, std::vector<FaceBox>&output)
{
	std::sort(input.begin(),input.end(),
            [](const FaceBox& a, const FaceBox&b) {
			return a.score > b.score;  
			});

	int box_num=input.size();

	std::vector<int> merged(box_num,0);

	for(int i=0;i<box_num;i++)
	{ 
		if(merged[i])
			continue;

		output.push_back(input[i]);

		float h0=input[i].y1-input[i].y0+1;
		float w0=input[i].x1-input[i].x0+1;
		float area0=h0*w0;

		for(int j=i+1;j<box_num;j++)
		{
			if(merged[j]) 
				continue;

			float inner_x0=std::max(input[i].x0,input[j].x0);
			float inner_y0=std::max(input[i].y0,input[j].y0);

			float inner_x1=std::min(input[i].x1,input[j].x1);
			float inner_y1=std::min(input[i].y1,input[j].y1);

			float inner_h=inner_y1-inner_y0+1;
			float inner_w=inner_x1-inner_x0+1;

			if(inner_h<=0 || inner_w<=0)
				continue;

			float inner_area=inner_h*inner_w;
			float h1=input[j].y1-input[j].y0+1;
			float w1=input[j].x1-input[j].x0+1;
			float area1=h1*w1;
			float score;

			if(type == NMS_UNION)
			{
				score=inner_area/(area0+area1-inner_area);
			}
			else
			{
				score=inner_area/std::min(area0,area1);
			}

			if(score>threshold)
				merged[j]=1;
		}
	}
}

void regress_boxes(std::vector<FaceBox>& rects)
{
	for(unsigned int i=0;i<rects.size();i++)
	{
        FaceBox& box=rects[i];

		float h=box.y1-box.y0+1;
		float w=box.x1-box.x0+1;

		box.x0=box.x0+w*box.regress[0];
		box.y0=box.y0+h*box.regress[1];
		box.x1=box.x1+w*box.regress[2];
		box.y1=box.y1+h*box.regress[3];
    }
}

void square_boxes(std::vector<FaceBox>& rects)
{
	for(unsigned int i=0;i<rects.size();i++)
	{
		float h=rects[i].y1-rects[i].y0+1;
		float w=rects[i].x1-rects[i].x0+1;
		float l=std::max(h,w);

		rects[i].x0=rects[i].x0+(w-l)*0.5;
		rects[i].y0=rects[i].y0+(h-l)*0.5;
		rects[i].x1=rects[i].x0+l-1;
		rects[i].y1=rects[i].y0+l-1;
	}
}

void padding(int img_h, int img_w, std::vector<FaceBox>& rects)
{
	for(unsigned int i=0; i<rects.size();i++)
	{
		rects[i].px0=std::max(rects[i].x0,1.0f);
		rects[i].py0=std::max(rects[i].y0,1.0f);
		rects[i].px1=std::min(rects[i].x1,(float)img_w);
		rects[i].py1=std::min(rects[i].y1,(float)img_h);
	}
} 

void generate_bounding_box(const float * confidence_data,
        const float * reg_data, float scale, float threshold,
        int feature_h, int feature_w, std::vector<FaceBox> &output, bool transposed)
{
    int stride = 2;
    int cellSize = 12;
    int img_h= feature_h;
    int img_w = feature_w;
    int count=img_h*img_w;
    if (!confidence_data) {
        return;
    }
    confidence_data += count;

    for (int i = 0; i<count; i++){
        if (*(confidence_data + i) >= threshold){
            int y = i / img_w;
            int x = i - img_w * y;

            float top_x = (float)((x * stride) / scale);
            float top_y = (float)((y * stride) / scale);
            float bottom_x = (float)((x * stride + cellSize - 1) / scale);
            float bottom_y = (float)((y * stride + cellSize - 1) / scale);

            FaceBox box;
            box.x0 = top_x;
            box.y0 = top_y;
            box.x1 = bottom_x;
            box.y1 = bottom_y;

            box.score = *(confidence_data + i);

            int c_offset=y*img_w+x;
            int c_size=img_w*img_h;

            if(transposed)
            {

                box.regress[1]=reg_data[c_offset];
                box.regress[0]=reg_data[c_offset+c_size];
                box.regress[3]=reg_data[c_offset+2*c_size];
                box.regress[2]= reg_data[c_offset+3*c_size];
            }
            else {

                box.regress[0]=reg_data[c_offset];
                box.regress[1]=reg_data[c_offset+c_size];
                box.regress[2]=reg_data[c_offset+2*c_size];
                box.regress[3]= reg_data[c_offset+3*c_size];
            }

            output.push_back(box);
        }
    }

}

void process_boxes(std::vector<FaceBox>& input, int img_h, int img_w, std::vector<FaceBox>& rects)
{
    nms_boxes(input,0.6,NMS_UNION,rects);
	regress_boxes(rects);
	square_boxes(rects);
	padding(img_h,img_w,rects);
} 

void  cal_pyramid_list(int height, int width, int min_size, float factor,std::vector<ScaleWindow>& list)
{ 
	int min_side = std::min(height, width);
	double m = 12.0 / min_size;
	min_side=min_side*m;
	double cur_scale=1.0;
	double scale;

	while (min_side >= 12)
	{
		scale=m*cur_scale;
		cur_scale=cur_scale *factor; 
		min_side *= factor;

		int hs = std::ceil(height*scale);
		int ws = std::ceil(width*scale);

        ScaleWindow win;
		win.h=hs;
		win.w=ws;
		win.scale=scale;
		list.push_back(win);
	}
}

void set_input_buffer(std::vector<cv::Mat>& input_channels,
		float* input_data, const int height, const int width) 
{
	for (int i = 0; i < 3; ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}
}


/*  测试用，编译时能跑出正确结果再上传。  */
#ifdef TEST
#include <sys/time.h>
main(int argc, char* argv[])
{   
    vector<string> image_read;
    vector<string> image_save;
    string path_read = "/media/the/D4FA828F299D817A/img_release";
    string path_save = "./test/best";
    GetFileNames(path_read, image_read);
    GetSaveNames(path_read, path_save, image_save);

    // std::string image_name="test.jpg";
    std::string model_path;
    int res;
    while((res = getopt(argc, argv, "f:m:")) != -1)
    {
        switch(res)
        {
            // case 'f':
            //     image_name = optarg;
            //     break;
            case 'm':
                model_path = optarg;
                break;
            default:
                break;
        }
    }

    // init tengine
    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    if(request_tengine_version("0.9") < 0)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }
    detector det;

    if(det.load_model(model_path))
    {
        std::cout << "load model failed!\n";
        return 1;
    }

     std::vector<face_box> det_faces;
    double alltime;
    for(int picnum = 0; picnum < image_read.size(); picnum++)
    {
        cout<< picnum << ": " << image_read[picnum] <<endl;
        cv::Mat img = cv::imread(image_read[picnum]);
        if(img.empty())
        {
            std::cout << "image open error: " << image_read[picnum] << std::endl;
            return 1;
        }
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        det.detect(img, det_faces);
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        alltime += mytime;
        std::cout << "--------------------------------------\n";
        std::cout << "run time is " << mytime << " ms\n";

        for(int i = 0; i < ( int )det_faces.size(); i++)
        {
            
            face_box box = det_faces[i];
            
            cv::rectangle(img, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
                        2);

            std::ostringstream score_str;
            score_str.precision(3);
            score_str << box.score;
            std::string label = score_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
            cv::rectangle(img,
                        cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 0), CV_FILLED);
            cv::putText(img, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
            // printf("Face %d:\t%.0f%%\t", i, box.score * 100);
            // printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
        }
        det_faces.clear();
        cv::imwrite(image_save[picnum],img);
        // cv::imshow("ret",img);
        // cv::waitKey(0);
    }
    
    std::cout << "run all time is " << alltime << " ms\n";
    det.release();
}
#endif