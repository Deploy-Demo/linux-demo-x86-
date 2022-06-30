#!/usr/bin/bash

PROJ_DIR="$1"
INCLUDE_DIR="${PROJ_DIR}/libs/model_infer/include"
LIB_DIR="${PROJ_DIR}/libs/model_infer/lib"
DEPLOY_DIR="${PROJ_DIR}/libs/PaddleX-release-2.0.0/deploy/cpp"
OPENCV_DIR="${DEPLOY_DIR}/deps/opencv3.4.6gcc4.8ffmpeg"

ln -sf "${OPENCV_DIR}/include" "${INCLUDE_DIR}/opencv"
ln -sf "${OPENCV_DIR}/lib" "${LIB_DIR}/opencv"
ln -sf "${DEPLOY_DIR}/build/lib/libmodel_infer.so" "${LIB_DIR}/libmodel_infer.so"
ln -sf "${PROJ_DIR}/libs/yaml-cpp/src/ext-yaml-cpp/include" "${INCLUDE_DIR}/yaml-cpp"
mv "${PROJ_DIR}/libs/yaml-cpp/lib/libyaml-cpp.so.0.6.2" "${PROJ_DIR}/libs/yaml-cpp/lib/libyaml-cpp.so.0.6"
ln -sf "${PROJ_DIR}/libs/yaml-cpp/lib/libyaml-cpp.so.0.6" "${LIB_DIR}/libyaml-cpp.so.0.6"

ln -sf "${LIB_DIR}"/*.so* /usr/lib/
rename 's/.so.3.4.6/.so.3.4/' "${LIB_DIR}/opencv"/*.so.3.4.6
ln -sf "${LIB_DIR}/opencv"/*.so.3.4 /usr/lib/
