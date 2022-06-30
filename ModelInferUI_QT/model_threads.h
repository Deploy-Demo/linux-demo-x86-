#ifndef MODEL_THREADS_H
#define MODEL_THREADS_H
#include <QThread>
#include <QList>
#include <QMutex>
#include "data.h"
#include "model_deploy/common/include/model_infer.h"

// ----------------------初始化线程----------------------
// 使用qthread做线程控制的好处：可以利用信号和槽，实现异步函数的效果(子线程中间或结束，实时让主线程响应)
class ThreadModelInit : public QThread
{
    Q_OBJECT
public:
    ThreadModelInit(MetaInformation _metaInfo);
    void ClearAllModels();
    void run() override;
private:
    MetaInformation metaInfo;


signals:
    void init_finished(QList<ModelWrapper*> models, QString modelType);
};


// ----------------------Infer线程----------------------
class ThreadModelInfer : public QThread
{
    Q_OBJECT
public:
    ThreadModelInfer(MetaInformation _metaInfo, QList<ModelWrapper*> _models, QList<std::string> _imagePaths);
    void run() override;
private:
    // data
    MetaInformation metaInfo;
    QList<ModelWrapper*> models;
    QList<std::string> imagePaths;

signals:
    // 信号
    void warmup_finished(double elapse);
    void batch_finished(double elapse);
    void infer_finished(double elapse);
};

// ----------------------infer线程----------------------
class ThreadSingleInfer : public QThread
{
    Q_OBJECT
public:
    ThreadSingleInfer(MetaInformation _metaInfo, ModelWrapper* _model, std::string _imgpath, const cv::Mat& _src);
    void run() override;
private:
    // func
    void EnqueueResult();
    // data
    MetaInformation metaInfo;
    ModelWrapper* model;
    std::string imgpath;
    cv::Mat src;
signals:
    void single_finished(double elapse);  // 推理结束，返回结果
};

// ----------------------存图线程----------------------
class ThreadSaveResult : public QThread
{
    Q_OBJECT
public:
    ThreadSaveResult();
    void run() override;
    bool quit = false;
    QMutex mutex;
private:
signals:

};

#endif // MODEL_THREADS_H
