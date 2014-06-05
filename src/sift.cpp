#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

int main(int argc, char* argv[])
{
    char* filename = argv[1];
    Mat image = imread(filename);
    Mat imageGray = imread(filename, 0);
    Mat descriptors;
    vector<KeyPoint> keypoints;
    initModule_nonfree();

    Ptr<Feature2D> sift1 = Algorithm::create<Feature2D>("Feature2D.SIFT");
    sift1->set("contrastThreshold", 0.01f);
    (*sift1)(imageGray, noArray(), keypoints, descriptors);

    for (int i = 0; i < keypoints.size(); i++)
    {
        Point temp = keypoints[i].pt;
        circle(image, temp, 3, Scalar(255,0,0));
    }


    imshow("Get SIFT of Image", image);

    waitKey();
    
    return 0;
}
