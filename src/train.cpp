#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

const string svmsDir = "/svms";
const string bowImageDescriptorsDir = "/bowImageDescriptors";
const string vocabularyFile = "vocabulary.xml.gz";

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


void read_training_data(std::vector<Mat>& training_data,
                        std::vector<float>& labels,
                        char* input_file)
{
    std::ifstream infile(input_file);
    
    std::string line;
    while (std::getline(infile, line))
    {
        std::vector<std::string> x = split(line, ' ');
        const char* filename = x.at(0).c_str();
        int label = atoi(x.at(1).c_str());

        Mat image = imread(filename);
        training_data.push_back(image);
        labels.push_back((float) label);

    }
}


void train_svm(CvSVM& svm, std::vector<Mat>& training_data, std::vector<Mat>& labels,
               Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
               const string& resPath)
{
    /* first check if a previously trained svm for the current class has been saved to file */
    string svmFilename = resPath + svmsDir + "/" + objClassName + ".xml.gz";

    FileStorage fs( svmFilename, FileStorage::READ);
    if( fs.isOpened() )
    {
        cout << "*** LOADING SVM CLASSIFIER FOR CLASS " << objClassName << " ***" << endl;
        svm.load( svmFilename.c_str() );
    }
    else
    {
        cout << "*** TRAINING CLASSIFIER FOR CLASS " << objClassName << " ***" << endl;
        cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "..." << endl;

        // Get classification ground truth for images in the training set
        vector<ObdImage> images;
        vector<Mat> bowImageDescriptors;
        vector<char> objectPresent;
        vocData.getClassImages( objClassName, CV_OBD_TRAIN, images, objectPresent );

        // Compute the bag of words vector for each image in the training set.
        calculateImageDescriptors( images, bowImageDescriptors, bowExtractor, fdetector, resPath );

        // Remove any images for which descriptors could not be calculated
        removeEmptyBowImageDescriptors( images, bowImageDescriptors, objectPresent );

        CV_Assert( svmParamsExt.descPercent > 0.f && svmParamsExt.descPercent <= 1.f );
        if( svmParamsExt.descPercent < 1.f )
        {
            int descsToDelete = static_cast<int>(static_cast<float>(images.size())*(1.0-svmParamsExt.descPercent));

            cout << "Using " << (images.size() - descsToDelete) << " of " << images.size() <<
                    " descriptors for training (" << svmParamsExt.descPercent*100.0 << " %)" << endl;
            removeBowImageDescriptorsByCount( images, bowImageDescriptors, objectPresent, svmParamsExt, descsToDelete );
        }

        // Prepare the input matrices for SVM training.
        Mat trainData( (int)images.size(), bowExtractor->getVocabulary().rows, CV_32FC1 );
        Mat responses( (int)images.size(), 1, CV_32SC1 );

        // Transfer bag of words vectors and responses across to the training data matrices
        for( size_t imageIdx = 0; imageIdx < images.size(); imageIdx++ )
        {
            // Transfer image descriptor (bag of words vector) to training data matrix
            Mat submat = trainData.row((int)imageIdx);
            if( bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize() )
            {
                cout << "Error: computed bow image descriptor size " << bowImageDescriptors[imageIdx].cols
                     << " differs from vocabulary size" << bowExtractor->getVocabulary().cols << endl;
                exit(-1);
            }
            bowImageDescriptors[imageIdx].copyTo( submat );

            // Set response value
            responses.at<int>((int)imageIdx) = objectPresent[imageIdx] ? 1 : -1;
        }

        cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << endl;
        CvSVMParams svmParams;
        CvMat class_wts_cv;
        setSVMParams( svmParams, class_wts_cv, responses, svmParamsExt.balanceClasses );
        CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
        setSVMTrainAutoParams( c_grid, gamma_grid,  p_grid, nu_grid, coef_grid, degree_grid );
        svm.train_auto( trainData, responses, Mat(), Mat(), svmParams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid );
        cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << endl;

        svm.save( svmFilename.c_str() );
        cout << "SAVED CLASSIFIER TO FILE" << endl;
    }
}


int main(int argc, char** argv)
{
    if( argc != 3 && argc != 6 )
    {
        help(argv);
        return -1;
    }

    cv::initModule_nonfree();

    const string vocPath = argv[1], resPath = argv[2];

    // Read or set default parameters
    string vocName;
    DDMParams ddmParams;
    VocabTrainParams vocabTrainParams;
    SVMTrainParamsExt svmTrainParamsExt;

    makeUsedDirs( resPath );

    FileStorage paramsFS( resPath + "/" + paramsFile, FileStorage::READ );
    if( paramsFS.isOpened() )
    {
       readUsedParams( paramsFS.root(), vocName, ddmParams, vocabTrainParams, svmTrainParamsExt );
       CV_Assert( vocName == getVocName(vocPath) );
    }
    else
    {
        vocName = getVocName(vocPath);
        if( argc!= 6 )
        {
            cout << "Feature detector, descriptor extractor, descriptor matcher must be set" << endl;
            return -1;
        }
        ddmParams = DDMParams( argv[3], argv[4], argv[5] ); // from command line
        // vocabTrainParams and svmTrainParamsExt is set by defaults
        paramsFS.open( resPath + "/" + paramsFile, FileStorage::WRITE );
        if( paramsFS.isOpened() )
        {
            writeUsedParams( paramsFS, vocName, ddmParams, vocabTrainParams, svmTrainParamsExt );
            paramsFS.release();
        }
        else
        {
            cout << "File " << (resPath + "/" + paramsFile) << "can not be opened to write" << endl;
            return -1;
        }
    }

    // Create detector, descriptor, matcher.
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create( ddmParams.detectorType );
    Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( ddmParams.descriptorType );
    Ptr<BOWImgDescriptorExtractor> bowExtractor;
    if( !featureDetector || !descExtractor )
    {
        cout << "featureDetector or descExtractor was not created" << endl;
        return -1;
    }
    {
        Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( ddmParams.matcherType );
        if( !featureDetector || !descExtractor || !descMatcher )
        {
            cout << "descMatcher was not created" << endl;
            return -1;
        }
        bowExtractor = makePtr<BOWImgDescriptorExtractor>( descExtractor, descMatcher );
    }

    // Print configuration to screen
    printUsedParams( vocPath, resPath, ddmParams, vocabTrainParams, svmTrainParamsExt );
    // Create object to work with VOC
    VocData vocData( vocPath, false );

    // 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
    Mat vocabulary = trainVocabulary( resPath + "/" + vocabularyFile, vocData, vocabTrainParams,
                                      featureDetector, descExtractor );
    bowExtractor->setVocabulary( vocabulary );

    // 2. Train a classifier and run a sample query for each object class
    const vector<string>& objClasses = vocData.getObjectClasses(); // object class list
    for( size_t classIdx = 0; classIdx < objClasses.size(); ++classIdx )
    {
        // Train a classifier on train dataset
#if defined HAVE_OPENCV_OCL && _OCL_SVM_
        cv::ocl::CvSVM_OCL svm;
#else
        CvSVM svm;
#endif
        trainSVMClassifier( svm, svmTrainParamsExt, objClasses[classIdx], vocData,
                            bowExtractor, featureDetector, resPath );

        // Now use the classifier over all images on the test dataset and rank according to score order
        // also calculating precision-recall etc.
        computeConfidences( svm, objClasses[classIdx], vocData,
                            bowExtractor, featureDetector, resPath );
        // Calculate precision/recall/ap and use GNUPlot to output to a pdf file
        computeGnuPlotOutput( resPath, objClasses[classIdx], vocData );
    }
    return 0;
}
