#ifndef DATA_H
#define DATA_H
#include <string>

struct MetaInformation
{
    std::string model_type;
    std::string img_dir;
    std::string model_dir;
    int model_num = 1;
    int gpu_id = 0;
    int cycles = 1;
    int target_width = 512;
    int target_height = 512;
    bool warmup = true;
    bool warmuped = false;
    int warmup_cycles = 30;
    bool save_result = false;
    bool use_trt = true;
    int min_shape_w = 0;
    int min_shape_h = 0;
    int opt_shape_w = 0;
    int opt_shape_h = 0;
    int max_shape_w = 0;
    int max_shape_h = 0;
    int min_subgraph_size = 3;
    std::string precision = "fp32";
    std::string shape_range_info_path = "./shape_range_info.pbtxt";
    bool is_paddlex_model = false;
};
#endif // DATA_H
