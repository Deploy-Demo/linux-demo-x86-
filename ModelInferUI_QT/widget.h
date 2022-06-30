#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QMutex>
#include "data.h"
#include "model_threads.h"
#include "model_deploy/common/include/model_infer.h"


QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    // 构建布局函数
    void initUI();
    // 初始化信号和槽
    void initSignalSolots();
    // clear model
    void ClearAllModels();
    // 关闭事件
    void closeEvent(QCloseEvent* event);

private:
    Ui::Widget *ui;

    // 数据
    QStringList modeltypes; // 模型类型
    QStringList modelnums;  // 模型数量字符串
    QStringList pres;       // 精度类型字符串
    MetaInformation metaInfo;
    QList<ModelWrapper*> models;
    QList<std::string> imgPaths;

    ThreadModelInit* threadInit;    
    ThreadModelInfer* threadInfer;
    ThreadSaveResult* threadSave;

signals:
    void needUpdate();

private slots:
    void slot_needUpdate();

    void slot_button_imgdir_clicked();
    void slot_button_modeldir_clicked();
    void slot_button_start_clicked();
    void slot_button_visualize_clicked();
    void slot_button_updatemodel_clicked();

    void slot_combo_modeltype_currentIndexChanged(int index);
    void slot_combo_modelnum_currentIndexChanged(int index);
    void slot_combo_precision_currentIndexChanged(int index);

    void slot_lineEdit_imgdir_textChanged(const QString &text);
    void slot_lineEdit_modeldir_textChanged(const QString &text);
    void slot_lineEdit_gpuid_textChanged(const QString &text);
    void slot_lineEdit_cycles_textChanged(const QString &text);
    void slot_lineEdit_newW_textChanged(const QString &text);
    void slot_lineEdit_newH_textChanged(const QString &text);
    void slot_lineEdit_minshapeW_textChanged(const QString &text);
    void slot_lineEdit_minshapeH_textChanged(const QString &text);
    void slot_lineEdit_maxshapeW_textChanged(const QString &text);
    void slot_lineEdit_maxshapeH_textChanged(const QString &text);
    void slot_lineEdit_optshapeW_textChanged(const QString &text);
    void slot_lineEdit_optshapeH_textChanged(const QString &text);
    void slot_lineEdit_minsubgraph_textChanged(const QString &text);

    void slot_textEdit_log_textChanged(const QString &text);

    void slot_check_warmup_stateChanged(int);
    void slot_check_saveresult_stateChanged(int);
    void slot_check_usetrt_stateChanged(int);

    // thread related
    void slot_thread_init_finished(QList<ModelWrapper*> _models, QString _model_type);
    void slot_thread_warmup_finished(double avgt);
    void slot_thread_batch_finished(double elapse);
    void slot_thread_infer_finished(double avgt);
};
#endif // WIDGET_H
