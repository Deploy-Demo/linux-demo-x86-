﻿// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <string>
#include <vector>
#include "model_deploy/common/include/paddle_deploy.h"
#include "model_deploy/common/include/model_infer.h"

#include "model_deploy/common/include/logger.h"
#include "model_deploy/common/include/timer.h"

//新增跨平台需求
#ifdef _WIN32
char file_spator(){
    return '\\';
}
#else
char file_sepator(){
    return '/';
}
#endif


/*
* 模型初始化/注册接口
*
* model_type: 初始化模型类型: det,seg,clas,paddlex
*
* model_filename: 模型文件路径
*
* params_filename: 参数文件路径
*
* cfg_file: 配置文件路径
*
* use_gpu: 是否使用GPU
*
* gpu_id: 指定第x号GPU
*
* paddlex_model_type: model_type为paddlx时，返回的实际paddlex模型的类型: det, seg, clas
*/
extern "C" PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type)
{
	// [suliang] 确保mask模型的类型也为det，而不能传入mask否则初始化报错
	if (strcmp(model_type, "mask") == 0) model_type = "det";

	// create model
	PaddleDeploy::Model* model = PaddleDeploy::CreateModel(model_type);  //FLAGS_model_type
	LOGC("INFO", "create model ok");

	// model init
	model->Init(cfg_file);
	//LOGC("INFO", "init model ok(yaml,preprocess, postprocess)");

	// inference engine init
	PaddleDeploy::PaddleEngineConfig engine_config;
	engine_config.model_filename = model_filename;
	engine_config.params_filename = params_filename;
	engine_config.use_gpu = use_gpu;
	engine_config.gpu_id = gpu_id;

	bool init = model->PaddleEngineInit(engine_config);
	if (!init)
	{
		LOGC("ERR", "init model failed");
	}
	//else
	//{
	//	LOGC("INFO", "init model successfully: use_gpu=%d, gpu_id=%d, model path=%s", use_gpu, gpu_id, model_filename);
	//}

	// det, seg, clas, paddlex
	if (strcmp(model_type, "paddlex") == 0) // 是paddlex模型，则返回具体支持的模型类型: det, seg, clas
	{
		// detector
		if (model->yaml_config_["model_type"].as<std::string>() == std::string("detector"))
		{
			strcpy(paddlex_model_type, "det");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("segmenter"))
		{
			strcpy(paddlex_model_type, "seg");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("classifier"))
		{
			strcpy(paddlex_model_type, "clas");
		}
	}

	return model;
}

// 初始化模型带tensorRT加速
// [suliang] 2021-12-15 增加5个输入参数：min_input_shape, max_input_shape, optim_input_shape分别代表输入尺寸的输入范围， precision代表计算精度(0=fp32,1=fp16,2=int8),min_subgraph_size代表最小优化子图
extern "C" PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type,
	const int* min_input_shape, const int* max_input_shape, const int* optim_input_shape, int precision, int min_subgraph_size, int target_width, int target_height, const char* shape_range_info_path)
{
	LOGC("INFO", "enter InitModel_TRT func");

	// [suliang] 确保mask模型的类型也为det，而不能传入mask否则初始化报错
	if (strcmp(model_type, "mask") == 0) model_type = "det";

	// create model
	PaddleDeploy::Model* model = PaddleDeploy::CreateModel(model_type);  //FLAGS_model_type
	LOGC("INFO", "create model ok: %s", model_type);

	// model init
	model->Init(cfg_file);
	LOGC("INFO", "init model ok(yaml,preprocess, postprocess)");

	// inference engine init
	PaddleDeploy::PaddleEngineConfig engine_config;
	engine_config.model_filename = model_filename;
	engine_config.params_filename = params_filename;
	engine_config.use_gpu = use_gpu;
	engine_config.gpu_id = gpu_id;

	// 务必使能trt
	engine_config.use_trt = true;

	// 注意：根据优化目标需要手动调整
	engine_config.precision = precision;				// 精度选择，默认fp32,还有fp16,int8
	engine_config.min_subgraph_size = min_subgraph_size;// 最小子图，越大则优化度越低，越大越可能忽略动态图: 设置40+不报错但也没啥优化
	engine_config.max_workspace_size = 1 << 30;

	// [suliang]新增trt fine tune选项
	engine_config.target_width = target_width;
	engine_config.target_height = target_height;
	engine_config.shape_range_info_path = std::string(shape_range_info_path);
	engine_config.model_type = std::string(model_type);

	// 注意：根据模型和输入图像大小，需要手动调整如下变量
	//std::vector<int> min_input_shape = { 1, 3, 512, 512 };
	//std::vector<int> max_input_shape = { 1, 3, 1024, 1024 };
	//std::vector<int> optim_input_shape = { 1, 3, 1024, 1024 };
	std::vector<int> min_input_shape_ = { min_input_shape[0], min_input_shape[1], min_input_shape[2], min_input_shape[3] };
	std::vector<int> max_input_shape_ = { max_input_shape[0], max_input_shape[1], max_input_shape[2], max_input_shape[3] };
	std::vector<int> optim_input_shape_ = { optim_input_shape[0], optim_input_shape[1], optim_input_shape[2], optim_input_shape[3] };
	// 分别定义最小、最大、最优输入尺寸：需要根据模型输入尺寸调整
	// 这里三种模型输入的关键字不同(clas对应inputs, det对应image, seg对应x)，可通过netron查看INPUTS.name，比如seg模型INPUTS.name=x
	// 另外如果有动态输入尺寸不匹配的节点，需要手动定义
	if (strcmp("clas", model_type) == 0) {
		// Adjust shape according to the actual model
		engine_config.min_input_shape["inputs"] = min_input_shape_;
		engine_config.max_input_shape["inputs"] = max_input_shape_;
		engine_config.optim_input_shape["inputs"] = optim_input_shape_;
	}
	else if (strcmp("det", model_type) == 0) {
		// Adjust shape according to the actual model
		engine_config.min_input_shape["image"] = min_input_shape_;
		engine_config.max_input_shape["image"] = max_input_shape_;
		engine_config.optim_input_shape["image"] = optim_input_shape_;
	}
	else if (strcmp("seg", model_type) == 0) {
		// Additional nodes need to be added, pay attention to the output prompt
		engine_config.min_input_shape["x"] = min_input_shape_;
		engine_config.max_input_shape["x"] = max_input_shape_;
		engine_config.optim_input_shape["x"] = optim_input_shape_;
	}
	LOGC("INFO", "min input shape: %d, %d, %d, %d", min_input_shape_[0], min_input_shape_[1], min_input_shape_[2], min_input_shape_[3]);
	LOGC("INFO", "max input shape: %d, %d, %d, %d", max_input_shape_[0], max_input_shape_[1], max_input_shape_[2], max_input_shape_[3]);
	LOGC("INFO", "opt input shape: %d, %d, %d, %d", optim_input_shape_[0], optim_input_shape_[1], optim_input_shape_[2], optim_input_shape_[3]);


	bool init = model->PaddleEngineInit(engine_config);
	LOGC("INFO", "init paddle engine: %d", (int)init);
	if (!init)
	{
		LOGC("INFO", "init paddle engine failed");
	}
	//else
	//{
	//	LOGC("INFO", "init paddle engine success");
	//}
	
	// det, seg, clas, paddlex
	if (strcmp(model_type, "paddlex") == 0) // 是paddlex模型，则返回具体支持的模型类型: det, seg, clas
	{
		// detector
		if (model->yaml_config_["model_type"].as<std::string>() == std::string("detector"))
		{
			strcpy(paddlex_model_type, "det");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("segmenter"))
		{
			strcpy(paddlex_model_type, "seg");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("classifier"))
		{
			strcpy(paddlex_model_type, "clas");
		}
	}
	return model;
}


/*
* 检测推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include category_id£¬score£¬coordinate¡£
*
* nBoxesNum£º number of box
*
* LabelList: label list of result
*/
extern "C" void Det_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(unsigned char));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	// nBoxesNum[0] = results.size();  // results.size()得到的是batch_size
	nBoxesNum[0] = results[0].det_result->boxes.size();  // 得到单张图片预测的bounding box数
	std::string label = "";
	//std::cout << "res: " << results[num] << std::endl;
	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // 得到所有框的数据
	{
		//std::cout << "category: " << results[num].det_result->boxes[i].category << std::endl;
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // 类别的id
		// score
		output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // 得分
		//// box
		output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // 左上、右下的顶点
		output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* 分割推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include label_map
*/
extern "C" void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output)
{
	//LOGC("INFO", "seg in thread id [%d]", GetCurrentThreadId());
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
		//LOGC("INFO", "infer input img w=%d, h=%d, c=%d", nWidth, nHeight, nChannel);
	}
	else
	{
		//std::cout << "Only support 3 channel image." << std::endl;
		LOGC("ERR", "Only support 3 channel images, but got channels ", nChannel);
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(unsigned char));
	//cv::imwrite("D://modelinfercpp_275.bmp", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);
	// batch修改：这里应该会得到返回的每张图的label_map，那么下面就应该分别处理results中每张图对应的label_map
	std::vector<uint8_t> result_map = results[0].seg_result->label_map.data; // vector<uint8_t> -- 结果map
	//LOGC("INFO", "finish infer, with result_map length=%d", result_map.size());
	// 拷贝输出结果到输出上返回 -- 将vector<uint8_t>转成unsigned char *
	memcpy(output, &result_map[0], result_map.size() * sizeof(unsigned char));
}


/*
* 分割推理接口batch predict
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include label_map
*/
extern "C" void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output)
{
	std::vector<PaddleDeploy::Result> results;
	if (imgs.size() != output.size()) {
		LOGC("ERR", "image batch size(%d) not match with results size(%d)", imgs.size(), output.size());
	}
	// Read image
	int im_vec_size = imgs.size();
	std::vector<cv::Mat> im_vec;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		LOGC("ERR", "Only support 3 channel images, but got channels ", nChannel);
		return;
	}
	for (int i = 0; i < im_vec_size; i++) {
		cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
		memcpy(input.data, imgs[i], nHeight * nWidth * nChannel * sizeof(unsigned char));
		im_vec.emplace_back(std::move(input));
	}
	if (!model->Predict(im_vec, &results, 1)) {
		LOGC("ERR", "predict batch images failed");
	}
	// batch修改：这里应该会得到返回的每张图的label_map，那么下面就应该分别处理results中每张图对应的label_map
	for (int i = 0; i < im_vec_size; i++) {
		std::vector<uint8_t> result_map = results[i].seg_result->label_map.data; // vector<uint8_t> -- 结果map
		// 拷贝输出结果到输出上返回 -- 将vector<uint8_t>转成unsigned char *
		memcpy(output[i], &result_map[0], result_map.size() * sizeof(unsigned char));
	}
}




/*
* 识别推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* score: result of pridict ,include score
*
* category: result of pridict ,include category_string
*
* category_id: result of pridict ,include category_id
*/
extern "C" void Cls_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(unsigned char));
	cv::imwrite("D:\\1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	//LOGC("INFO", "begin predict");
	model->Predict(imgs, &results, 1);
	//LOGC("INFO", "got pred result: score=%f", results[0].clas_result->score);
	//LOGC("INFO", "got pred result: category_id=%d", results[0].clas_result->category_id);
	//LOGC("INFO", "got pred result: category=%s", results[0].clas_result->category);
	*category_id = results[0].clas_result->category_id;
	// 拷贝输出类别结果到输出上返回 -- string --> char* 
	memcpy(category, results[0].clas_result->category.c_str(), strlen(results[0].clas_result->category.c_str()));
	// 拷贝输出概率值返回
	*score = results[0].clas_result->score;
}


/*
* MaskRCNN推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* box_output: result of pridict ,include label+score+bbox
*
* mask_output: result of pridict ,include label_map
*
* nBoxesNum: result of pridict ,include BoxesNum
*
* LabelList: result of pridict ,include LabelList
*/
extern "C" void Mask_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	// [suliang] 增加一个mask_output的初始化全0操作，否则下面的mask操作时判断0的操作会导致内存中没有初始化为0的点被保存为map的一部分
	memset(mask_output, 0, nWidth * nHeight);

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(unsigned char));
	imgs.push_back(std::move(input));

	// predict  -- 多次点击单张推理时会出错
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);  // 在Infer处发生错误

	nBoxesNum[0] = results[0].det_result->boxes.size();  // 得到单张图片预测的bounding box数
	std::string label = "";

	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // 得到所有框的数据
	{
		// 边界框预测结果
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		box_output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // 类别的id
		// score
		box_output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // 得分
		//// box
		box_output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		box_output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // 左上、右下的顶点
		box_output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		box_output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];

		//Mask预测结果：这里有个坑，外部传入的Mask_必须初始化为全0，否则会导致mask_output[j]==0的判断忽略掉，从而引入额外噪点
		for (int j = 0; j < results[0].det_result->boxes[i].mask.data.size(); j++)
		{
			if (mask_output[j] == 0)
			{
				mask_output[j] = results[0].det_result->boxes[i].mask.data[j];
			}
		}
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* 模型销毁/注销接口
*/
extern "C" void DestructModel(PaddleDeploy::Model * model)
{
	if (model != NULL) {
		delete model;
		model = NULL;
	}
	if (model == NULL) LOGC("INFO", "destruct model success");
	else LOGC("ERR", "delete model failed");
}


// 新增二次封装：初始化
void ModelWrapper::InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, char* paddlex_model_type,
	const int* min_input_shape, const int* max_input_shape, const int* optim_input_shape, int precision, int min_subgraph_size, 
	int target_width, int target_height, const char* shape_range_info_path)
{
	LOGC("INFO", "enter InitModelEnter func");
	// 初始化线程池：创建指定个数线程，每个线程指定到线程池的一个线程号
	pool = new ThreadPool(num_threads);
	pool->init();

	std::string model_filename = std::string(model_dir) + std::string(1,file_sepator()) + "model.pdmodel";
	std::string params_filename = std::string(model_dir) + std::string(1,file_sepator()) + "model.pdiparams";
        std::string cfg_file = std::string(model_dir) + std::string(1,file_sepator());
        if (strcmp(model_type, "seg") == 0)
            cfg_file += "deploy.yaml";
        else if (strcmp(model_type, "det") == 0 || strcmp(model_type, "mask") == 0)
             cfg_file += "infer_cfg.yml";
        else
            cfg_file += "model.yml";

	bool use_gpu = true;
	//char* paddle_model_type = NULL;
	if (!use_trt) {
		_model = InitModel(model_type,
			model_filename.c_str(),    // *.pdmodel
			params_filename.c_str(),   // *.pdiparams
			cfg_file.c_str(),          // *.yaml 
			use_gpu,
			gpu_id,
			paddlex_model_type);
	}
	else
	{
		_model = InitModel_TRT(model_type,
			model_filename.c_str(),    // *.pdmodel
			params_filename.c_str(),   // *.pdiparams
			cfg_file.c_str(),          // *.yaml 
			use_gpu,
			gpu_id,
			paddlex_model_type,
			min_input_shape, max_input_shape, optim_input_shape, precision, min_subgraph_size,
			target_width, target_height, shape_range_info_path);
	}
}

// 新增二次封装：单图推理
void ModelWrapper::SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map)
{
	cv::Mat src;
	if (channels == 1) {
		src = cv::Mat(height, width, CV_8UC1, imageData);
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	}
	else
	{
		src = cv::Mat(height, width, CV_8UC3, imageData);
	}
	int predChannels = src.channels();
	unsigned char* _imageData = src.data;
	auto future1 = pool->submit(Seg_ModelPredict, _model, _imageData, width, height, predChannels, result_map);
	future1.get();
}

// 检测模型
void ModelWrapper::DetPredictEnter(unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList)
{
	cv::Mat src;
	if (channels == 1) {
		src = cv::Mat(height, width, CV_8UC1, imageData);
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	}
	else
	{
		src = cv::Mat(height, width, CV_8UC3, imageData);
	}
	int predChannels = src.channels();
	unsigned char* _imageData = src.data;
	auto future1 = pool->submit(Det_ModelPredict, _model, _imageData, width, height, predChannels, output, nBoxesNum, LabelList);
	future1.get();
}

// 分类模型
void ModelWrapper::ClsPredictEnter(unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id)
{
	cv::Mat src;
	if (channels == 1) {
		src = cv::Mat(height, width, CV_8UC1, imageData);
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	}
	else
	{
		src = cv::Mat(height, width, CV_8UC3, imageData);
	}
	int predChannels = src.channels();
	unsigned char* _imageData = src.data;
	auto future1 = pool->submit(Cls_ModelPredict, _model, _imageData, width, height, predChannels, score, category, category_id);
	future1.get();
}

// Mask模型
void ModelWrapper::MaskPredictEnter(unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
{
	cv::Mat src;
	if (channels == 1) {
		src = cv::Mat(height, width, CV_8UC1, imageData);
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	}
	else
	{
		src = cv::Mat(height, width, CV_8UC3, imageData);
	}
	int predChannels = src.channels();
	unsigned char* _imageData = src.data;
	auto future1 = pool->submit(Mask_ModelPredict, _model, _imageData, width, height, predChannels, box_output, mask_output, nBoxesNum, LabelList);
	future1.get();
}


// 新增二次封装：模型资源释放
void ModelWrapper::DestructModelEnter()
{
	// 释放线程池中所有线程
	pool->shutdown();
	if (pool != NULL) {
		delete pool;
		pool = NULL;
	}
	// 释放模型资源
	if (_model != NULL) {
		DestructModel(_model);
	}
}


// 新增二次封装接口api
extern "C" ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, char* paddlex_model_type,
	const int* min_input_shape, const int* max_input_shape, const int* optim_input_shape, const int precision, const int min_subgraph_size, 
	int target_width, int target_height, const char* shape_range_info_path)
{
	ModelWrapper* modelObj = new ModelWrapper();
	modelObj->InitModelEnter(model_type, model_dir, gpu_id, use_trt, paddlex_model_type, 
		min_input_shape, max_input_shape, optim_input_shape, precision, min_subgraph_size, target_width, target_height, shape_range_info_path);

	return modelObj;
}

//extern "C" ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt)
//{
//	ModelWrapper* modelObj = new ModelWrapper();
//	modelObj->InitModelEnter(model_type, model_dir, gpu_id, use_trt);
//	return modelObj;
//}


extern "C" void ModelObjDestruct(ModelWrapper * modelObj)
{
	// 先释放模型内部的资源
	modelObj->DestructModelEnter();
	// 再释放堆区模型资源
	delete  modelObj;
}

extern "C" void ModelObjPredict_Seg(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap)
{
	Timer timer;
	timer.start();
	modelObj->SegPredictEnter(imageData, width, height, channels, resultMap);
	LOGC("Info", "time of Total: %f", timer.stop_and_return());
}

extern "C" void ModelObjPredict_Det(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList)
{
	Timer timer;
	timer.start();
	modelObj->DetPredictEnter(imageData, width, height, channels, output, nBoxesNum, LabelList);
	LOGC("Info", "time of Total: %f", timer.stop_and_return());
}

extern "C" void ModelObjPredict_Cls(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id)
{
	Timer timer;
	timer.start();
	modelObj->ClsPredictEnter(imageData, width, height, channels, score, category, category_id);
	LOGC("Info", "time of Total: %f", timer.stop_and_return());
}

extern "C" void ModelObjPredict_Mask(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
{
	Timer timer;
	timer.start();
	modelObj->MaskPredictEnter(imageData, width, height, channels, box_output, mask_output, nBoxesNum, LabelList);
	LOGC("Info", "time of Total: %f", timer.stop_and_return());
}
