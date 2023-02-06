#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "reader.h"
#include "writer.h"
#include "value.h"
#include "ji_utils.h"
#include "SampleAlgorithm.hpp"

#define JSON_ALERT_FLAG_KEY ("is_alert")
#define JSON_ALERT_FLAG_TRUE true
#define JSON_ALERT_FLAG_FALSE false


SampleAlgorithm::SampleAlgorithm()
{    
}

SampleAlgorithm::~SampleAlgorithm()
{    
    UnInit();
}

STATUS SampleAlgorithm::Init()
{
    // 从默认的配置文件读取相关配置参数
    const char *configFile = "/usr/local/ev_sdk/config/algo_config.json";
    SDKLOG(INFO) << "Parsing configuration file: " << configFile;
    std::ifstream confIfs(configFile);
    if (confIfs.is_open())
    {
        size_t len = getFileLen(confIfs);
        char *confStr = new char[len + 1];
        confIfs.read(confStr, len);
        confStr[len] = '\0';
    	SDKLOG(INFO) << "Configs:"<<confStr;
        mConfig.ParseAndUpdateArgs(confStr);
        delete[] confStr;
        confIfs.close();
    }
    mDetector = std::make_shared<SampleDetector>();
    mDetector->Init();
    return STATUS_SUCCESS;
}


STATUS SampleAlgorithm::UnInit()
{
    if(mDetector.get() != nullptr)
    {
        SDKLOG(INFO) << "uninit";
        mDetector->UnInit();
        mDetector.reset();
    }
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::UpdateConfig(const char *args)
{
    if (args == nullptr)
    {
        SDKLOG(ERROR) << "mConfig string is null ";
        return ERROR_CONFIG;
    }
    mConfig.ParseAndUpdateArgs(args);
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::GetOutFrame(JiImageInfo **out, unsigned int &outCount)
{
    outCount = mOutCount;

    mOutImage[0].nWidth = mOutputFrame.cols;
    mOutImage[0].nHeight = mOutputFrame.rows;
    mOutImage[0].nFormat = JI_IMAGE_TYPE_BGR;
    mOutImage[0].nDataType = JI_UNSIGNED_CHAR;
    mOutImage[0].nWidthStride = mOutputFrame.step;
    mOutImage[0].pData = mOutputFrame.data;

    *out = mOutImage;
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::Process(const cv::Mat &inFrame, const char *args, JiEvent &event)
{
    //输入图片为空的时候直接返回错误
    if (inFrame.empty())
    {
        SDKLOG(ERROR) << "Invalid input!";
        return ERROR_INPUT;
    }
   
    //由于roi配置是归一化的坐标,所以输出图片的大小改变时,需要更新ROI的配置  
    if (inFrame.cols != mConfig.currentInFrameSize.width || inFrame.rows != mConfig.currentInFrameSize.height)
    {
	    SDKLOG(INFO)<<"Update ROI Info...";
        mConfig.UpdateROIInfo(inFrame.cols, inFrame.rows);
    }

    //如果输入的参数不为空且与上一次的参数不完全一致,需要调用更新配置的接口
    if(args != nullptr && mStrLastArg != args) 
    {	
    	mStrLastArg = args;
        SDKLOG(INFO) << "Update args:" << args;
        mConfig.ParseAndUpdateArgs(args);
        LOG(INFO)<<"Config mask_output_path:"<<mConfig.algoConfig.mask_output_path;
    }
    
    // 算法处理
    cv::Mat img = inFrame.clone();
    Mat mask = Mat::zeros(inFrame.rows, inFrame.cols, CV_8UC1);
    vector<Polygon> detectedPolygons;
    mDetector->ProcessImage(img, mask, detectedPolygons);    
    int row = inFrame.rows;
    int col = inFrame.cols;
    float over_rate = mConfig.algoConfig.over_rate;
    int num_1 = 0;
    Json::Value mound_ratios;

    if(mConfig.currentROIOrigPolygons.size() >= 1)
    { 
        for(int i = 0; i < mConfig.currentROIOrigPolygons.size(); ++i) {
            int num_1 = 0;
            int num_area = 0;
            //std::cout << mConfig.currentROIOrigPolygons[i] << " ";

            cv::Mat drawing_contour = cv::Mat::zeros(inFrame.size(), CV_8UC1);
	    std::vector<std::vector<cv::Point>> conptsss;
	    std::vector<cv::Point> roipts = mConfig.currentROIOrigPolygons[i];//三角形或四边形的顶点，排序应为顺时针或者逆时针排序
 
	    conptsss.push_back(roipts);
	    cv::drawContours(drawing_contour, conptsss, 0, cv::Scalar(255), cv::FILLED);
 
            std::vector<cv::Point> rectanglepts;//三角形或四边形内的所有点的坐标
	    for (int ii = 0; ii < row; ii++)
	    {
		for (int jj = 0; jj < col; jj++)
		{
                        if  (drawing_contour.ptr<uchar>(ii)[jj] == 255){
                               num_area = num_area + 1;           
                        }
			if (drawing_contour.ptr<uchar>(ii)[jj] == 255 &&  mask.at<uchar>(ii,jj) == 1){
				rectanglepts.push_back(cv::Point(jj, ii));
                                num_1 = num_1 + 1;
			}
		}
	    }

        float mound_ratio = (float)num_1 / (float)num_area;//(col*row);
        mound_ratios.append(mound_ratio);
    //LOG(INFO)<<"ratio:"<<num_1<<";"<<col<<";"<<row<<";"<<mound_ratio;
    }}    

    bool isNeedAlert = false;
    for(int k = 0; k < mound_ratios.size(); k++)
    {
        if (mound_ratios[k] >= over_rate){
            isNeedAlert = true; }// 是否需要报警
    }
    // 创建输出图
    if(mConfig.algoConfig.mask_output_path.length()>0)
    {
        imwrite(mConfig.algoConfig.mask_output_path, mask);   ///自动测试，生成单通道mask图，mask图里的每个像素值表示类别索引号
        inFrame.copyTo(mOutputFrame);
    }
    else
    {
        cv::Mat rgb_img = mDetector->generate_rgb(inFrame, mask, mConfig);  ///封装sdk，生成RGB染色图
        float ratio = mConfig.fill_color[3];
        //LOG(INFO)<<"*****:"<<mConfig.fill_color<<ratio;
        addWeighted(inFrame, 1-ratio, rgb_img, ratio, 0, mOutputFrame);
    }
    
    // 画ROI区域
    if (mConfig.drawROIArea && !mConfig.currentROIOrigPolygons.empty())
    {
        drawPolygon(mOutputFrame, mConfig.currentROIOrigPolygons, cv::Scalar(mConfig.roiColor[0], mConfig.roiColor[1], mConfig.roiColor[2]),
                    mConfig.roiColor[3], cv::LINE_AA, mConfig.roiLineThickness, mConfig.roiFill);
    }

    if (isNeedAlert && mConfig.drawWarningText)
    {
        drawText(mOutputFrame, mConfig.warningTextMap[mConfig.language], mConfig.warningTextSize,
                 cv::Scalar(mConfig.warningTextFg[0], mConfig.warningTextFg[1], mConfig.warningTextFg[2]),
                 cv::Scalar(mConfig.warningTextBg[0], mConfig.warningTextBg[1], mConfig.warningTextBg[2]), mConfig.warningTextLeftTop);
    }

    //for(int i = 0; i < mConfig.currentROIOrigPolygons.size(); ++i) {
    //    std::cout <<"hhh"<< mConfig.currentROIOrigPolygons[i][0] << " ";
    //}
    //[0, 0; 1920, 0; 1920, 1080; 0, 1080] [0, 0; 1920, 0; 1920, 1080; 0, 1080]    
    //LOG(INFO)<<"ROI:"<<mConfig.currentROIOrigPolygons;
    // 将结果封装成json字符串
    bool jsonAlertCode = JSON_ALERT_FLAG_FALSE;
    if (isNeedAlert)
    {
        jsonAlertCode = JSON_ALERT_FLAG_TRUE;
    }
    Json::Value jRoot;
    Json::Value jAlgoValue;
    Json::Value jDetectValue;
    
    jAlgoValue[JSON_ALERT_FLAG_KEY] = jsonAlertCode;    
    //Json::Value mound_ratios;
    //mound_ratios.append(mound_ratio);
    jAlgoValue["ratios"] = mound_ratios;
    jRoot["algorithm_data"] = jAlgoValue;
    
    //create model data
    jDetectValue["mask"] = mConfig.algoConfig.mask_output_path;
    jRoot["model_data"] = jDetectValue;

    Json::StreamWriterBuilder writerBuilder;
    writerBuilder.settings_["precision"] = 2;
    writerBuilder.settings_["emitUTF8"] = true;    
    std::unique_ptr<Json::StreamWriter> jsonWriter(writerBuilder.newStreamWriter());
    std::ostringstream os;
    jsonWriter->write(jRoot, &os);
    mStrOutJson = os.str();    
    // 注意：JI_EVENT.code需要根据需要填充，切勿弄反
    if (isNeedAlert)
    {
        event.code = JISDK_CODE_ALARM;
    }
    else
    {
        event.code = JISDK_CODE_NORMAL;
    }
    event.json = mStrOutJson.c_str();    
    return STATUS_SUCCESS;
}

