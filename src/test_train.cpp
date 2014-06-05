#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <vector>
#include "catch.hpp"
#include "train.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE( "Load images from input file", "[read_training_data]" ) {
    std::vector<cv::Mat> trainingData;
    std::vector<float> labels;
    char* inputFile = "unit-test.labels";
    read_training_data(trainingData, labels, inputFile);
    REQUIRE( trainingData.size() == 10 );
    REQUIRE( labels.size() == 10 );
    for (int i = 0; i <= 9; i++)
        REQUIRE( labels.at(i) == i + 1 );
}
