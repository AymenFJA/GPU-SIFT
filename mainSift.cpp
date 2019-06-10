/********************************************************/
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
// //              celle @ csc.kth.se                        //
// //             Improved By Aymen Alsaadi                  //
// //             aymen.alsaadi@rutgers.edu                  //
// //********************************************************//
#include <iostream>
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include "cudaImage.h"
#include "cudaSift.h"
#include <vector>
#include <stdlib.h>  // for strtol
int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{    
  int devNum = 0, imgSet = 0;
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
  //check for valid GEOTIFF input 
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
  //Tiling Method @aymenalsaadi :) 
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

     
     cv::Mat limg = cv::Mat(src_limg, cv::Rect(src_x1,src_y1,src_x2,src_y2));
     cv::Mat rimg = cv::Mat(trg_rimg, cv::Rect(trg_x1,trg_y1,trg_x2,trg_y2)); 
     //cv::gpu::GpuMat gm;    
     //gm.upload(src_limg);
     //gm.upload(trg_rimg);
    
     //cv::gpu::GpuMat rimg= trg_rimg;
     //cv::gpu::GpuMat.upload(limg);
     //cv::gpu::GpuMat.upload(rimg);
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  unsigned int w2 = rimg.cols;
  unsigned int h2 = rimg.rows;
 
  std::string command1 ,command2;
  std::string cmd,cmd2="gdalinifo ";
  std::cout << "gdal source image 1 inforamtion    " <<system(cmd.c_str())<<std::endl;

 
  std::cout << "Source image size = (" << w << "," << h << ")" << std::endl;
  std::cout << "Target image size = (" << w2 << "," << h2 << ")" << std::endl;

  //@aymen.alsaadi This funcion will use adaptive threshold instead for giving a static value //
  //
  //
  //  //cv::Mat dst2,dst1,limg2,rimg2;
  //    //limg2=cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  //      //cv::adaptiveThreshold(limg2, dst1, 200,cv::ADAPTIVE_THRESH_GAUSSIAN_C , cv::THRESH_BINARY, 65, 0);
  //        //cv::imwrite("/home/aymen/SummerRadical/GPU-SIFT/dst1.jpg", dst1);
  //          //rimg2=cv::imread(argv[2],CV_LOAD_IMAGE_GRAYSCALE);
  //            //cv::adaptiveThreshold(rimg2, dst2, 200, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 65, 0);
  //              //cv::imwrite("/home/aymen/SummerRadical/GPU-SIFT/dst2.jpg", dst2);
  //                //*************************************************************************************************************//
  // Initial Cuda images and download images to device

  std::cout << "Initializing data..." << std::endl;
  InitCuda(devNum); 
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download(); 

  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 1.0f;
  float thresh =1.3f;
  std::cout<<"Threshold value :"<<thresh<<std::endl;
  InitSiftData(siftData1, 1000000, true, true);
  InitSiftData(siftData2, 1000000, true, true);
  
  
  // A bit of benchmarking 
  //for (int thresh1=1.00f;thresh1<=4.01f;thresh1+=0.50f) {
  float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    for (int i=0;i<1000;i++) {
      ExtractSift(siftData1, img1, 3, initBlur, thresh, 0.0f, false, memoryTmp);
      ExtractSift(siftData2, img2, 3, initBlur, thresh, 0.0f, false, memoryTmp);
    }
    FreeSiftTempMemory(memoryTmp);
    
    // Match Sift features and find a homography
    for (int i=0;i<1;i++)
      MatchSiftData(siftData1, siftData2);
    float homography[9];
    int numMatches;
    FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
    int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 1.3);
    
    std::cout << "Number features detected by SIFT descriptors : " <<"Source Image "<<  siftData1.numPts << ", Target Image " << siftData2.numPts << std::endl;
    std::cout << "Number of Matched features: " << numMatches <<std::endl;
    std::cout << "Number of fitted features after applying ImproveHomography : " << numFit <<std::endl;
    std::cout << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " <<std::endl;

    //}
  
  // Print out and store summary data
  PrintMatchData(siftData1, siftData2, img1);
  PrintMatchData(siftData1, siftData2, img2);
  
  cv::imwrite("/home/aymen/SummerRadical/SIFT-GPU/source_output.pgm", limg);
  std::cout<< "img1 saved"<<std::endl;
  cv::imwrite("/home/aymen/SummerRadical/SIFT-GPU/target_output.pgm", rimg);
  std::cout<< "img2 saved"<<std::endl;
  std::cout << "Output Images are saved in the same directory !) "<<std::endl;


  //MatchAll(siftData1, siftData2, homography);
  
  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
#if 1
  homography[0] = homography[4] = -1.0f;
  homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
  homography[2] = 1279.0f;
  homography[5] = 959.0f;
#endif
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << " " << sift1[i].xpos << " " << sift1[i].ypos << std::endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++) 
	sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err<100.0f) // 100.0
	found = true;
      if (err<100.0f || j==sift1[i].match) { // 100.0
	if (j==sift1[i].match && err<100.0f)
	  std::cout << " *";
	else if (j==sift1[i].match) 
	  std::cout << " -";
	else if (err<100.0f)
	  std::cout << " +";
	else
	  std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " " << (int)dx << " " << (int)dy << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
  std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
  std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
  std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img)
{
  //@aymenalsaadi creating a csv file to save the matched keypoints
  std::ofstream myfile;
  myfile.open ("CUDA_data_matches.csv", std::ofstream::out | std::ofstream::trunc);
  myfile<<"x1, y1, x2, y2,dx,dy"<<std::endl;
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (sift1[j].match_error<5) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
      //@aymenalsaadi writing the matched keypoints (x1,y1-x2,y2) into csv file
      myfile<<((int)sift1[j].xpos)<<","<<((int)sift1[j].ypos)<<","<<((int)sift2[k].xpos)<<","<<((int)sift2[k].ypos)<<std::endl;
#if 0
      if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
	std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
	std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
	std::cout << "scale=" << sift1[j].scale << "  ";
	std::cout << "error=" << (int)sift1[j].match_error << "  ";
	std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
	std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
      }
#endif
#if 1
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[j].xpos + dx*l/len);
	int y = (int)(sift1[j].ypos + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }
#endif
    }
    int x = (int)(sift1[j].xpos+0.5);
    int y = (int)(sift1[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}


