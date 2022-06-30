QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    model_threads.cpp \
    widget.cpp

HEADERS += \
    _dirent.h \
    data.h \
    fileio.h \
    model_threads.h \
    transforms.h \
    widget.h

FORMS += \
    widget.ui

# -----------------model_infer-----------------
# 头文件在本地，so文件放到系统路径下，放在/usr/local/lib会报找不到(因为部分机器的/usr/local不是系统路径)，放在/usr/lib则能直接找到
INCLUDEPATH += ModelInferUI_QT/libs/model_infer/include
LIBS += /usr/lib/libmodel_infer*.so

# -----------------opencv-----------------
INCLUDEPATH += ModelInferUI_QT/libs/model_infer/include/opencv
LIBS += /usr/lib/libopencv_*.so.3.4

# -----------------yaml-cpp-----------------
INCLUDEPATH += ModelInferUI_QT/libs/model_infer/include/yaml-cpp
LIBS += /usr/lib/libyaml-cpp.so.0.6

# -----------------tensorRT-----------------
# 把trt的库拷贝到/usr/lib理论上一定找得到，但发现没成功；最后是在sudo gedit /etc/ld.so.conf文件中添加trt/lib文件夹然后ldconfig更新，然后也要如下2句引入才找到
INCLUDEPATH += TensorRT/include
LIBS += -L/usr/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser

# LIBS += /usr/lib/libdnnl.so.2

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
