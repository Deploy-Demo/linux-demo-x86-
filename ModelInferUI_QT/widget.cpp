#include "widget.h"
#include "ui_widget.h"
#include <QFileDialog>
#include <QSpacerItem>
#include <QMessageBox>
#include "fileio.h"
#include "opencv2/opencv.hpp"
#include "transforms.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    initUI();
    initSignalSolots();
}

void Widget::ClearAllModels()
{
    if(models.size() > 0){
        for(int i=0;i<models.size();i++)
            if(models[i]!=nullptr) {
                ModelObjDestruct(models[i]);
                models[i] = nullptr;
            }
    }
    models.clear();
}

void Widget::initUI()
{
    QString _modeltypes = ",clas,seg,det,mask,paddlex";
    modeltypes = _modeltypes.split(",");
    for(int i=0;i<modeltypes.size();i++) ui->combo_modeltype->addItem(modeltypes[i]);

    QString _modelnums = "1,2,3,4,5,6,7,8,9,10";
    modelnums = _modelnums.split(",");
    for(int i=0;i<modelnums.size();i++) ui->combo_modelnum->addItem(modelnums[i]);
    QString _pres = "fp32,fp16,int8";
    pres = _pres.split(",");
    for(int i=0;i<pres.size();i++) ui->combo_precision->addItem(pres[i]);

    ui->button_start->setEnabled(false);
    ui->button_updatemodel->setEnabled(false);

    ui->button_visualize->setVisible(false);

    // 创建存图线程
//    threadSave->run();
}

// 初始化信号和槽的连接
void Widget::initSignalSolots()
{
    connect(ui->button_imgdir, SIGNAL(clicked()), this, SLOT(slot_button_imgdir_clicked()));
    connect(ui->button_modeldir, SIGNAL(clicked()), this, SLOT(slot_button_modeldir_clicked()));
    connect(ui->button_start, SIGNAL(clicked()), this, SLOT(slot_button_start_clicked()));
    connect(ui->button_visualize, SIGNAL(clicked()), this, SLOT(slot_button_visualize_clicked()));
    connect(ui->button_updatemodel, SIGNAL(clicked()), this, SLOT(slot_button_updatemodel_clicked()));

    // 注意：连接时必须写上形参类型
    connect(ui->combo_modeltype, SIGNAL(currentIndexChanged(int)), this, SLOT(slot_combo_modeltype_currentIndexChanged(int)));
    connect(ui->combo_modelnum, SIGNAL(currentIndexChanged(int)), this, SLOT(slot_combo_modelnum_currentIndexChanged(int)));
    connect(ui->combo_precision, SIGNAL(currentIndexChanged(int)), this, SLOT(slot_combo_precision_currentIndexChanged(int)));

    connect(ui->lineEdit_imgdir, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_imgdir_textChanged(const QString&)));
    connect(ui->lineEdit_modeldir, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_modeldir_textChanged(const QString&)));
    connect(ui->lineEdit_gpuid, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_gpuid_textChanged(const QString&)));
    connect(ui->lineEdit_cycles, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_cycles_textChanged(const QString&)));
    connect(ui->lineEdit_newH, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_newH_textChanged(const QString&)));
    connect(ui->lineEdit_newW, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_newW_textChanged(const QString&)));
    connect(ui->lineEdit_minshapeH, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_minshapeH_textChanged(const QString&)));
    connect(ui->lineEdit_minshapeW, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_minshapeW_textChanged(const QString&)));
    connect(ui->lineEdit_maxshapeH, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_maxshapeH_textChanged(const QString&)));
    connect(ui->lineEdit_maxshapeW, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_maxshapeW_textChanged(const QString&)));
    connect(ui->lineEdit_optshapeH, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_optshapeH_textChanged(const QString&)));
    connect(ui->lineEdit_optshapeW, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_optshapeW_textChanged(const QString&)));
    connect(ui->lineEdit_minsubgraph, SIGNAL(textChanged(const QString&)), this, SLOT(slot_lineEdit_minsubgraph_textChanged(const QString&)));

    connect(ui->textEdit_log, SIGNAL(textChanged(const QStrig&)), this, SLOT(slot_textEdit_log_textChanged(const QString&)));

    connect(ui->check_warmup, SIGNAL(stateChanged(int)), this, SLOT(slot_check_warmup_stateChanged(int)));
    connect(ui->check_usetrt, SIGNAL(stateChanged(int)), this, SLOT(slot_check_usetrt_stateChanged(int)));
    connect(ui->check_saveresult, SIGNAL(stateChanged(int)), this, SLOT(slot_check_saveresult_stateChanged(int)));

    connect(this, SIGNAL(needUpdate()), this, SLOT(slot_needUpdate()));
}

void Widget::closeEvent(QCloseEvent* event)
{
    auto temp = QMessageBox::information(this, "tooltip", QString::fromLocal8Bit("Are you sure to close app?"), QMessageBox::Yes | QMessageBox::No);
    if (temp == QMessageBox::Yes)
    {
        // 检查模型是否释放
        if(models.size()>0){
            for(int i=0;i<models.size();i++){
                ModelObjDestruct(models[i]);
                models[i] = nullptr;
            }
            models.clear();
        }
        //
        event->accept();
    }
    else
    {
        event->ignore();
    }
}


void Widget::slot_needUpdate()
{
    ui->button_start->setEnabled(false);
    ui->button_updatemodel->setEnabled(true);
    metaInfo.warmuped = false;
}

// 更新模型
void Widget::slot_button_updatemodel_clicked()
{
    // clear all models before create new models
    ClearAllModels();

    // 子线程对模型初始化
    if(metaInfo.use_trt){
        ui->textEdit_log->insertPlainText(QString("[Info] Creating trt models will take a long long time, please wait with patient....\n"));
        ui->textEdit_log->insertPlainText(QString("[Info] Collecting trt model shape range info and save to %1....\n").arg(QString::fromStdString(metaInfo.shape_range_info_path)));
    }


    if (metaInfo.model_type == "paddlex"){
        metaInfo.is_paddlex_model = true;
    }
    if (metaInfo.is_paddlex_model){
        metaInfo.model_type = "paddlex";
    }

    threadInit = new ThreadModelInit(metaInfo);
    connect(threadInit, SIGNAL(finished()), threadInit, SLOT(deleteLater()));

    // 线程没有创建前无法绑定，所以需要在这里绑定
    connect(threadInit, SIGNAL(init_finished(QList<ModelWrapper*>, QString)),
            this, SLOT(slot_thread_init_finished(QList<ModelWrapper*>, QString)), Qt::BlockingQueuedConnection);

    threadInit->start();
    ui->button_updatemodel->setEnabled(false);
}

// start predict:  warmup->predict
void Widget::slot_button_start_clicked()
{
    ui->button_start->setEnabled(false);

    threadInfer = new ThreadModelInfer(metaInfo, models, imgPaths);
    connect(threadInfer, SIGNAL(finished()), threadInfer, SLOT(deleteLater()));
    connect(threadInfer, SIGNAL(warmup_finished(double)),
            this, SLOT(slot_thread_warmup_finished(double)), Qt::BlockingQueuedConnection);
    connect(threadInfer, SIGNAL(batch_finished(double)),
            this, SLOT(slot_thread_batch_finished(double)), Qt::BlockingQueuedConnection);
    connect(threadInfer, SIGNAL(infer_finished(double)),
            this, SLOT(slot_thread_infer_finished(double)), Qt::BlockingQueuedConnection);

    if(!metaInfo.warmuped){
        ui->textEdit_log->insertPlainText(QString("[Info] Start warmup %1 models with %2 warmup cycles.\n").arg(
                                              QString::asprintf("%d",models.size()), QString::asprintf("%d",metaInfo.warmup_cycles)));
    }
    threadInfer->start();
}

// 可视化
void Widget::slot_button_visualize_clicked()
{

}

// 选择图片文件夹
void Widget::slot_button_imgdir_clicked()
{
    QString imgDirPath = QFileDialog::getExistingDirectory(this, "Choose img directory", "/");
    if (imgDirPath.length() > 0)
    {
        ui->lineEdit_imgdir->setText(imgDirPath);
        ui->textEdit_log->insertPlainText(QString("Choose img dir: %1.\n").arg(imgDirPath));
        metaInfo.img_dir = imgDirPath.toStdString();
    }
    // get filenames
    for_each_file(metaInfo.img_dir,
        [&](const char* path, const char* name) {
            auto full_path = std::string(path).append({ file_sepator() }).append(name);
            std::string lower_name = tolower1(name);

            if (end_with(lower_name, ".jpg") || end_with(lower_name, ".jepg")) {
                imgPaths.push_back(full_path);
            }
            return false;
        }, true);
    ui->textEdit_log->insertPlainText(QString("Got %1 image files.\n").arg(imgPaths.size()));
}

// 选择模型文件夹
void Widget::slot_button_modeldir_clicked()
{
    QString modelDirPath = QFileDialog::getExistingDirectory(this, "Choose model directory", "/");
    if (modelDirPath.length() > 0)
    {
        ui->lineEdit_modeldir->setText(modelDirPath);
        ui->textEdit_log->insertPlainText(QString("Choose model dir: %1.\n").arg(modelDirPath));
        metaInfo.model_dir = modelDirPath.toStdString();

        // 调整按钮状态
        emit needUpdate();
    }
}

// 选择模型类型
void Widget::slot_combo_modeltype_currentIndexChanged(int index)
{
    QString mtype = modeltypes[index];
    ui->textEdit_log->insertPlainText(QString("Choose model type: %1.\n").arg(mtype));
    metaInfo.model_type = mtype.toStdString();
    metaInfo.is_paddlex_model = false;

    emit needUpdate();
}

// 选择模型数量
void Widget::slot_combo_modelnum_currentIndexChanged(int index)
{
    QString numstr = modelnums[index];
    ui->textEdit_log->insertPlainText(QString("Choose model num: %1.\n").arg(numstr));
    metaInfo.model_num = numstr.toInt();

    emit needUpdate();
}

void Widget::slot_combo_precision_currentIndexChanged(int index)
{
    QString pre = pres[index];
    ui->textEdit_log->insertPlainText(QString("Choose precision: %1.\n").arg(pre));
    metaInfo.precision = pre.toStdString();

    emit needUpdate();
}

void Widget::slot_lineEdit_imgdir_textChanged(const QString &text)
{
    ui->textEdit_log->insertPlainText(QString("Choose new imgdir: %1.\n").arg(text));
    metaInfo.img_dir = text.toStdString();
}

void Widget::slot_lineEdit_modeldir_textChanged(const QString &text)
{
    ui->textEdit_log->insertPlainText(QString("Choose new modeldir: %1.\n").arg(text));
    metaInfo.model_dir = text.toStdString();
}

void Widget::slot_lineEdit_gpuid_textChanged(const QString &text)
{
    int gpuid = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input gpu: %1.\n").arg(gpuid));
    metaInfo.gpu_id = gpuid;

    emit needUpdate();
}

void Widget::slot_lineEdit_cycles_textChanged(const QString &text)
{
    int cycle = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input cycle: %1.\n").arg(cycle));
    metaInfo.cycles = cycle;
}

void Widget::slot_lineEdit_newW_textChanged(const QString &text)
{
    int newW = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input new Width: %1.\n").arg(newW));
    metaInfo.target_width = newW;

    emit needUpdate();
}

void Widget::slot_lineEdit_newH_textChanged(const QString &text)
{
    int newH = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input new Height: %1.\n").arg(newH));
    metaInfo.target_height = newH;

    emit needUpdate();
}

void Widget::slot_lineEdit_minshapeW_textChanged(const QString &text)
{
    int minW = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input min shape Width: %1.\n").arg(minW));
    metaInfo.min_shape_w = minW;

    emit needUpdate();
}

void Widget::slot_lineEdit_minshapeH_textChanged(const QString &text)
{
    int minH = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input min shape Height: %1.\n").arg(minH));
    metaInfo.min_shape_h = minH;

    emit needUpdate();
}

void Widget::slot_lineEdit_maxshapeW_textChanged(const QString &text)
{
    int maxW = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input max shape Width: %1.\n").arg(maxW));
    metaInfo.max_shape_w = maxW;

    emit needUpdate();
}

void Widget::slot_lineEdit_maxshapeH_textChanged(const QString &text)
{
    int maxH = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input max shape Height: %1.\n").arg(maxH));
    metaInfo.max_shape_h = maxH;

    emit needUpdate();
}

void Widget::slot_lineEdit_optshapeW_textChanged(const QString &text)
{
    int optW = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input opt shape Width: %1.\n").arg(optW));
    metaInfo.opt_shape_w = optW;

    emit needUpdate();
}

void Widget::slot_lineEdit_optshapeH_textChanged(const QString &text)
{
    int optH = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input opt shape Height: %1.\n").arg(optH));
    metaInfo.opt_shape_h = optH;

    emit needUpdate();
}

void Widget::slot_lineEdit_minsubgraph_textChanged(const QString &text)
{
    int minSubSize = text.toInt();
    ui->textEdit_log->insertPlainText(QString("Input min subgraph size: %1.\n").arg(minSubSize));
    metaInfo.min_subgraph_size = minSubSize;

    emit needUpdate();
}

void Widget::slot_textEdit_log_textChanged(const QString &text)
{
    ui->textEdit_log->moveCursor(QTextCursor::End);
}

void Widget::slot_check_warmup_stateChanged(int state)
{
    if(state == Qt::Checked){
        ui->textEdit_log->insertPlainText(QString("choosed warmup!\n"));
        metaInfo.warmup = true;
    }
    else if(state == Qt::Unchecked){
        ui->textEdit_log->insertPlainText(QString("Not choosed warmup!\n"));
        metaInfo.warmup = false;
    }
}

void Widget::slot_check_saveresult_stateChanged(int state)
{
    if(state == Qt::Checked){
        ui->textEdit_log->insertPlainText(QString("choosed save result!\n"));
        metaInfo.save_result = true;
    }
    else if(state == Qt::Unchecked){
        ui->textEdit_log->insertPlainText(QString("Not choosed save result!\n"));
        metaInfo.save_result = false;
    }
}

void Widget::slot_check_usetrt_stateChanged(int state)
{
    if(state == Qt::Checked){
        ui->textEdit_log->insertPlainText(QString("[Info] chosen use-trt!\n"));
        metaInfo.use_trt = true;
    }
    else if(state == Qt::Unchecked){
        ui->textEdit_log->insertPlainText(QString("[Info] Not chosen use-trt!\n"));
        metaInfo.use_trt = false;
    }

    emit needUpdate();
}

void Widget::slot_thread_init_finished(QList<ModelWrapper*> _models, QString _model_type)
{
    metaInfo.warmuped = false;
    metaInfo.model_type = _model_type.toStdString();
    models = _models;
    int sizes = _models.size();
    if(metaInfo.use_trt){
        ui->textEdit_log->insertPlainText(QString("[Info] Finished create %1 models with tensorRT.\n").arg(sizes));
    }else
    {
        ui->textEdit_log->insertPlainText(QString("[Info] Finished create %1 models.\n").arg(sizes));
    }

    ui->button_start->setEnabled(true);
    ui->button_updatemodel->setEnabled(false);
}

void Widget::slot_thread_warmup_finished(double elapse)
{
    // warmup done
    metaInfo.warmuped = true;
    double avgt = elapse / metaInfo.warmup_cycles / metaInfo.model_num;
    ui->textEdit_log->insertPlainText(QString("[Info] Finished warmup %1 models with %2 ms, average elaspe %3 ms.\n").arg(
                                          QString::asprintf("%d", models.size()),
                                          QString::asprintf("%f", elapse),
                                          QString::asprintf("%.0f", avgt)));

}

void Widget::slot_thread_batch_finished(double elapse)
{
    ui->textEdit_log->insertPlainText(QString("[Info] Finished infer %1 images with %2 ms.\n").arg(
                                        QString::asprintf("%d", models.size()),
                                        QString::asprintf("%f", elapse)));

}

void Widget::slot_thread_infer_finished(double elapse)
{
    ui->textEdit_log->insertPlainText(QString("[Info] Total imags: %1, Total elapse: %2 ms, average time: %3 ms.\n").arg(
                                          QString::asprintf("%d", imgPaths.size()),
                                          QString::asprintf("%f", elapse),
                                          QString::asprintf("%.0f", elapse/imgPaths.size())));
    ui->textEdit_log->insertPlainText(QString("[Info] ----------------------------------\n"));
    ui->button_start->setEnabled(true);
}


Widget::~Widget()
{
    delete ui;
}

