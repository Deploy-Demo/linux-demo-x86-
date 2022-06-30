#pragma once
#include "paddle_deploy.h"
#include "logger.h" // [suliang] LOGC
#include "thread_pool.h"


// ���߳�Ĭ������api������ԭ��model_infer.cpp�޸����βκͷ���ֵ����model��Ϊ�������
extern "C" PaddleDeploy::Model * InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type);
extern "C" PaddleDeploy::Model * InitModel_TRT(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type,
	const int* min_input_shape, const int* max_input_shape, const int* optim_input_shape, int precision = 0, int min_subgraph_size = 40, int target_width=512, int target_height=512, const char* shape_range_info_path=NULL);
extern "C" void DestructModel(PaddleDeploy::Model * model);
// �ָ�
extern "C" void Seg_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output);
extern "C" void Seg_ModelBatchPredict(PaddleDeploy::Model * model, const std::vector<unsigned char*> imgs, int nWidth, int nHeight, int nChannel, std::vector<unsigned char*> output);
// ���
extern "C" void Det_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList);
// ����
extern "C" void Cls_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id);
// Mask
extern "C" void Mask_ModelPredict(PaddleDeploy::Model * model, const unsigned char* img, int nWidth, int nHeight, int nChannel, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);


// ���Ӷ��η�װ�ࣺ��ģ�ͺ��̳߳ط�װ
class ModelWrapper
{
public:
	// ģ�ͳ�ʼ��
	void InitModelEnter(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, char* paddle_model_type = NULL, 
		const int* min_input_shape = NULL, const int* max_input_shape = NULL, const int* optim_input_shape = NULL, int precision = 0, int min_subgraph_size = 40,
		int target_width = 512, int target_height = 512, const char* shape_range_info_path = NULL);
	// �ָ�ģ�͵�ͼ����
	void SegPredictEnter(unsigned char* imageData, int width, int height, int channels, unsigned char* result_map);
	// ���ģ��
	void DetPredictEnter(unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
	// ����ģ��
	void ClsPredictEnter(unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
	// Maskģ��
	void MaskPredictEnter(unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);
	// ģ����Դ�ͷ�
	void DestructModelEnter();

private:
	ThreadPool* pool = NULL;
	PaddleDeploy::Model* _model = NULL;
	int num_threads = 1;
};

// ���Ӷ��̶߳��η�װ�ӿ�
extern "C" ModelWrapper * ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, char* paddlex_model_type = NULL,
	const int* min_input_shape = NULL, const int* max_input_shape = NULL, const int* optim_input_shape = NULL, const int precision = 0, const int min_subgraph_size = 40, 
	int target_width = 512, int target_height = 512, const char* shape_range_info_path = NULL);
extern "C" void ModelObjDestruct(ModelWrapper * modelObj);
extern "C" void ModelObjPredict_Seg(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap);
extern "C" void ModelObjPredict_Det(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
extern "C" void ModelObjPredict_Cls(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
extern "C" void ModelObjPredict_Mask(ModelWrapper * modelObj, unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);