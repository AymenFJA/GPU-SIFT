#include <iostream>
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include <vector>
#include <stdlib.h>  // for strtol
#include <string>
using std::string;


int main(int argc, char **argv)
{
  int devNum = 2, imgSet = 0;
  if( argc != 11)
    {
     std::cout <<"Usage :";
     std::cout <<"./cudaSift img1 x1 y1 x2 y2 img2 x1 y1 x2 y2 " << std::endl;
     std::cout <<" img1 : The source image to be matched! " << std::endl;
     std::cout <<" img2 : The target image to be matched! " << std::endl;
     std::cout <<" x1,y1,x2,y2 are the coordinates of the tile! " << std::endl;


     return -1;
    }



cv::Mat src_limg, trg_rimg;
  cv::imread(argv[1], 0).convertTo(src_limg, CV_32FC1);
  cv::imread(argv[6], 0).convertTo(trg_rimg, CV_32FC1);

if(! src_limg.data || ! trg_rimg.data)
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
  else
     {

  std::cout << "GEOTIFF image1 detected  " <<argv[1]<<std::endl;
  std::cout << "GEOTIFF image2 detected  " <<argv[6]<<std::endl;

     }


int src_x1= strtol(argv[2],NULL, 10);
  int src_y1= strtol(argv[3],NULL, 10);
  int src_x2= strtol(argv[4],NULL, 10);
  int src_y2= strtol(argv[5],NULL, 10);
  int trg_x1= strtol(argv[7],NULL, 10);
  int trg_y1= strtol(argv[8],NULL, 10);
  int trg_x2= strtol(argv[9],NULL, 10);
  int trg_y2= strtol(argv[10],NULL,10);

if (((src_x1 > src_limg.cols) || (src_y1 > src_limg.rows)) && ((src_x2 > src_limg.cols) ||(src_y2 > src_limg.rows)))
     {
      std::cout <<  "Requested tile size is bigger than the original source image size" << std::endl ;
      return -1;
     }
  else if (((trg_x1 > trg_rimg.cols) || (trg_y1 > trg_rimg.rows)) && ((trg_y2 > trg_rimg.cols) ||(trg_y2 > trg_rimg.rows)))
     {
     std::cout <<  "Requested tile size is bigger than the original target image size" << std::endl ;
     return -1;
     }
}
