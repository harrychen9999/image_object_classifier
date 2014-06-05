#ifndef TRAIN_HPP
#define TRAIN_HPP
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>





void read_training_data(std::vector<cv::Mat>& training_data,
                        std::vector<float>& labels,
                        char* input_file);



#endif
