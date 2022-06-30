#ifndef FILEIO_H
#define FILEIO_H

//
// 采用dirent.h进行文件、文件夹操作，该dirent.h可兼容windows/linux/unix
//
#include <functional>
#include <algorithm>
#include <string.h>
#ifdef _WIN32 || WIN32
#include "_dirent.h"
#else
#include <dirent.h>
#endif
#include <vector>
#include <iostream>

//----------------------------------------------文件、文件夹---------------------------------------------------
//#define throw_if(expression) if(expression)throw "error"
// 判断是否是文件夹
inline bool is_folder(const char* dir_name) {
    if (nullptr == dir_name)
        std::cout << "dir_name is nullprt";
    //throw_if(nullptr==dir_name);
    auto dir = opendir(dir_name);
    if (dir) {
        closedir(dir);
        return true;
    }
    return false;
}
#ifdef _WIN32 || WIN32
inline char file_sepator() {
    return '\\';
}
#else
inline char file_sepator() {
    return '/';
}
#endif
// 判断是否是文件夹
inline bool is_folder(const std::string& dir_name) {
    if (dir_name.empty())
        std::cerr << "dir_name is empty" << std::endl;
    return is_folder(dir_name.data());
}


/* 筛选文件夹内所有文件
 * 列出指定目录的所有文件(不包含目录)，对每个文件执行filter过滤器，
 * filter返回true时将文件名全路径加入std::vector
 * sub为true时为目录递归
 * 返回每个文件的全路径名
*/
using file_filter_type = std::function<bool(const char*, const char*)>;
static  std::vector<std::string> for_each_file(const std::string& dir_name, file_filter_type filter, bool sub = false) {
    std::vector<std::string> v;
    auto dir = opendir(dir_name.data());
    struct dirent* ent;
    if (dir) {
        while ((ent = readdir(dir)) != nullptr) {
            auto p = std::string(dir_name).append({ file_sepator() }).append(ent->d_name);
            if (sub) {
                if (0 == strcmp(ent->d_name, "..") || 0 == strcmp(ent->d_name, ".")) {
                    continue;
                }
                else if (is_folder(p)) {
                    auto r = for_each_file(p, filter, sub);
                    v.insert(v.end(), r.begin(), r.end());
                    continue;
                }
            }
            if (sub || !is_folder(p))//如果是文件，则调用过滤器filter
                if (filter(dir_name.data(), ent->d_name))
                    v.emplace_back(p);
        }
        closedir(dir);
    }
    return v;
}

//----------------------------------------------字符串---------------------------------------------------
//字符串大小写转换
inline std::string tolower1(const std::string& src) {
    auto dst = src;
    std::transform(src.begin(), src.end(), dst.begin(), ::tolower);
    return dst;
}

// 判断src是否以指定的字符串(suffix)结尾
inline bool end_with(const std::string& src, const std::string& suffix) {
    return src.substr(src.size() - suffix.size()) == suffix;
}

// 字符串替换
inline std::string replace_all_distinct(std::string str, const std::string old_value, const std::string new_value)
{
    for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
    {
        if ((pos = str.find(old_value, pos)) != std::string::npos)
        {
            str.replace(pos, old_value.length(), new_value);
        }
        else { break; }
    }
    return   str;
}

// 去除首尾空格
inline std::string trim(std::string s)
{
    if (!s.empty())
    {
        s.erase(0,s.find_first_not_of(" "));
        s.erase(s.find_last_not_of(" ") + 1);
    }
    return s;
}


#endif // FILEIO_H
