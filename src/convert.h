#ifndef CONVERT_H
#define CONVERT_H
#include<stdio.h>
#include<opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <XnOpenNI.h>
#include <XnCppWrapper.h>
#include "convert.h"


void depthMap_to_mat( const XnDepthPixel* depthMap,  cv::Mat& cv_image, int rows, int cols )
{
  /*
  cv_image = cv::Mat(rows, cols, CV_16UC1);
  cv::MatIterator_<ushort> it = cv_image.cv::Mat::begin<ushort>();
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      (*it) = depthMap[i*cols + j];
      it;
    }
    }*/
  // cv_image.create( rows, cols, CV_16UC1 );
  // memcpy(cv_image.data, depthMap, rows*cols*sizeof(CV_16UC1) );
  int sizes[2] = {rows, cols };
  cv_image  =  cv::Mat(2, sizes , CV_16UC1 ,  (void*)depthMap);

}

void imageMap_to_mat(const XnRGB24Pixel * imageMap,cv::Mat& cv_image, int rows, int cols)
{
  int sizes[2] = {rows, cols};
  cv_image  =  cv::Mat(2, sizes,CV_8UC3 ,  (void*)imageMap);

}

void IRMap_to_mat(const XnIRPixel * infraredMap, cv::Mat& cv_image, int rows, int cols  )
{
  int sizes[2] = {rows, cols};
  cv_image = cv::Mat(2, sizes, CV_16UC1, (unsigned char*)infraredMap);

}
#endif
