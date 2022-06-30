#include "model_threads.h"
#include "transforms.h"
#include "fileio.h"
#include <algorithm>
#include <QMutexLocker>

#define MAXBOXNUM 100
#define MAXSTRLEN 50

//---------------------------thread init--------------------------------------
ThreadModelInit::ThreadModelInit(MetaInformation _metaInfo)
{
    metaInfo = _metaInfo;
}


void ThreadModelInit::run()
{
    QStringList pres = {"fp32", "fp16", "int8"};
    char paddlex_model_type[10]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' '};
    int min_shape[4] = {1, 3, metaInfo.min_shape_h, metaInfo.min_shape_w};
    int max_shape[4] = {1, 3, metaInfo.max_shape_h, metaInfo.max_shape_w};
    int opt_shape[4] = {1, 3, metaInfo.opt_shape_h, metaInfo.opt_shape_w};
    int precision_code = pres.indexOf(QString::fromStdString(metaInfo.precision));
    QList<ModelWrapper*> _models = {};
    for(int i=0; i<metaInfo.model_num; i++){
        if(!metaInfo.use_trt){
            ModelWrapper* model = ModelObjInit(metaInfo.model_type.c_str(), metaInfo.model_dir.c_str(), metaInfo.gpu_id, metaInfo.use_trt, paddlex_model_type);
            _models.push_back(model);
        }else{
            ModelWrapper* model = ModelObjInit(metaInfo.model_type.c_str(), metaInfo.model_dir.c_str(), metaInfo.gpu_id, metaInfo.use_trt, paddlex_model_type,
                                               min_shape, max_shape, opt_shape, precision_code, metaInfo.min_subgraph_size,
                                               metaInfo.target_width, metaInfo.target_height, metaInfo.shape_range_info_path.c_str());
            _models.push_back(model);
        }
        if(metaInfo.model_type == "paddlex"){
            metaInfo.model_type = trim(std::string(paddlex_model_type));
        }
    }
    emit init_finished(_models, QString::fromStdString(metaInfo.model_type));
}


//---------------------------thread infer--------------------------------------
// single infer in main thread, then main thread can get signal from single thread, but single thread need blocking wait, which is not accepted by main thread.
// single infer in model infer thread, can not get each single infer, but can get every round infer.
ThreadModelInfer::ThreadModelInfer(MetaInformation _metaInfo, QList<ModelWrapper*> _models, QList<std::string> _imagePaths)
{
    metaInfo = _metaInfo;
    models = _models;
    imagePaths = _imagePaths;
}

void ThreadModelInfer::run()
{
    // 预热
    Timer timer;
    double elapse = 0.0;
    if(metaInfo.warmup && !metaInfo.warmuped){
        int warmup_cycles = metaInfo.warmup_cycles;
        cv::Mat src = cv::imread(imagePaths[0], cv::IMREAD_COLOR);
        float scale_factor;
        rescale(src, src, metaInfo.target_width, metaInfo.target_height, scale_factor);

        timer.start();
        for(int i=0; i<warmup_cycles; i++){
            for(int j = 0;j< models.size(); j++){
                if(metaInfo.model_type == "clas")
                {
                    float score;
                    char category[50];
                    int category_id;
                    ModelObjPredict_Cls(models[j], src.data, metaInfo.target_width, metaInfo.target_height, 3, &score, category, &category_id);
                }
                else if(metaInfo.model_type == "det")
                {
                    std::unique_ptr<float[]> output(new float[MAXBOXNUM * 6]);
                    int nbox;
                    std::unique_ptr<char[]> label_list(new char[MAXBOXNUM * MAXSTRLEN]);
                    ModelObjPredict_Det(models[j], src.data, metaInfo.target_width, metaInfo.target_height, 3, output.get(), &nbox, label_list.get());
                }
                else if(metaInfo.model_type == "seg")
                {
                    std::unique_ptr<unsigned char[]> resultMap(new unsigned char[metaInfo.target_width * metaInfo.target_height]);
                    ModelObjPredict_Seg(models[j], src.data, metaInfo.target_width, metaInfo.target_height, 3, resultMap.get());
//                    cv::Mat map = cv::Mat(metaInfo.target_height, metaInfo.target_width, CV_8UC1, resultMap.get());
//                    cv::imwrite("src.bmp", src);
//                    cv::imwrite("map.bmp", map);
                }
                else if(metaInfo.model_type == "mask")
                {
                    std::unique_ptr<float[]> box_output(new float[MAXBOXNUM * 6]);
                    std::unique_ptr<unsigned char[]>mask_output(new unsigned char[metaInfo.target_width * metaInfo.target_height]);
                    int nbox;
                    std::unique_ptr<char[]> label_list(new char[MAXBOXNUM * MAXSTRLEN]);
                    ModelObjPredict_Mask(models[j], src.data, metaInfo.target_width, metaInfo.target_height, 3, box_output.get(), mask_output.get(), &nbox, label_list.get());
                }
                else
                {
                    return;
                }
            }
        }
        elapse = timer.stop_and_return();
/*        double avg = elapse / warmup_cycles / models.size()*/;
        emit warmup_finished(elapse);
    }

    // infer
    QList<ThreadSingleInfer*> singleInfers;
    elapse = 0.0;
    for(int i=0;i<metaInfo.cycles;i++){
        for(int j=0;j<imagePaths.size();j++){
            cv::Mat src = cv::imread(imagePaths[j], cv::IMREAD_COLOR);
            float scale_factor;
            rescale(src, src, metaInfo.target_width, metaInfo.target_height, scale_factor);
            // predict
            timer.start();
            int idx = j % metaInfo.model_num;
            ThreadSingleInfer* t_infer = new ThreadSingleInfer(metaInfo, models[idx], imagePaths[j], src);
            connect(t_infer, SIGNAL(finished()), t_infer, SLOT(deleteLater()));

            singleInfers.push_back(t_infer);
            // models match with img or last of img
            if((idx == models.size() - 1) || (j == imagePaths.size()-1)){
                // parallel runing single infers
                for(int k=0;k<singleInfers.size();k++){
                    singleInfers[k]->start();
                }
                // wait parallel running single infers finish.
                for(int k=0;k<singleInfers.size();k++){
                    singleInfers[k]->wait();
                }
                singleInfers.clear();
            }
            double batchtime = timer.stop_and_return();
            emit batch_finished(batchtime);
            elapse += batchtime;
        }
    }
    emit infer_finished(elapse);
}


//---------------------------single infer--------------------------------------
ThreadSingleInfer::ThreadSingleInfer(MetaInformation _metaInfo, ModelWrapper* _model, std::string _imgpath, const cv::Mat& _src=cv::Mat())
{
    metaInfo = _metaInfo;
    model = _model;
    imgpath = _imgpath;
    if (_src.empty())
        src = cv::imread(imgpath, cv::IMREAD_COLOR);
    else
        src = _src;
}

void ThreadSingleInfer::run()
{
    Timer timer;
    timer.start();
    if(metaInfo.model_type == "clas")
    {
        float score;
        char category[50];
        int category_id;
        ModelObjPredict_Cls(model, src.data, metaInfo.target_width, metaInfo.target_height, 3, &score, category, &category_id);
    }
    else if(metaInfo.model_type == "det")
    {
        std::unique_ptr<float[]> output(new float[MAXBOXNUM * 6]);
        int nbox;
        std::unique_ptr<char[]> labelList(new char[MAXBOXNUM * MAXSTRLEN]);
        ModelObjPredict_Det(model, src.data, metaInfo.target_width, metaInfo.target_height, 3, output.get(), &nbox, labelList.get());
    }
    else if(metaInfo.model_type == "seg")
    {
        std::unique_ptr<unsigned char[]> resultMap(new unsigned char[metaInfo.target_width * metaInfo.target_height]);
        ModelObjPredict_Seg(model, src.data, metaInfo.target_width, metaInfo.target_height, 3, resultMap.get());

    }
    else if(metaInfo.model_type == "mask")
    {
        std::unique_ptr<float[]> box_output(new float[MAXBOXNUM * 6]);
        std::unique_ptr<unsigned char[]>mask_output(new unsigned char[metaInfo.target_width * metaInfo.target_height]);
        int nbox;
        std::unique_ptr<char[]> label_list(new char[MAXBOXNUM * MAXSTRLEN]);
        ModelObjPredict_Mask(model, src.data, metaInfo.target_width, metaInfo.target_height, 3, box_output.get(), mask_output.get(), &nbox, label_list.get());
    }
    else
    {
        return;
    }
    double elapse = timer.stop_and_return();
    emit single_finished(elapse);
}


//---------------------------thread save--------------------------------------
ThreadSaveResult::ThreadSaveResult()
{

}

void ThreadSaveResult::run()
{
    while(true){


        mutex.lock();

        mutex.unlock();
        if(quit) return;
    }
}
