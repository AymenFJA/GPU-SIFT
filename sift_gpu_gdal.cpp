
/* gdal_image.cpp -- Load GIS data into OpenCV Containers using the Geospatial Data Abstraction Library
*/
//*************************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil     //
//                 celle @ csc.kth.se                        //
//                Added functionality by                    //
//                      Aymen Alsaadi                      //
//                aymen.alsaadi@rutgers.edu               //
//*******************************************************//
#include <iostream>
#include <cmath>
#include <iomanip>
// OpenCV Headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include "cudaImage.h"
#include "cudaSift.h"
#include <vector>
#include <stdlib.h>  // for strtol
#include <string>
using std::string;
#include <stdexcept>


using namespace std;

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img, const string& s, const string& s2);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);


// define the corner points
//    Note that GDAL library can natively determine this
cv::Point2d tl( -122.441017, 37.815664 );
cv::Point2d tr( -122.370919, 37.815311 );
cv::Point2d bl( -122.441533, 37.747167 );
cv::Point2d br( -122.3715,   37.746814 );

// determine dem corners
cv::Point2d dem_bl( -122.0, 38);
cv::Point2d dem_tr( -123.0, 37);

// range of the heat map colors
std::vector<std::pair<cv::Vec3b,double> > color_range;


// List of all function prototypes
cv::Point2d lerp( const cv::Point2d&, const cv::Point2d&, const double& );

cv::Vec3b get_dem_color( const double& );

cv::Point2d world2dem( const cv::Point2d&, const cv::Size&);

cv::Point2d pixel2world( const int&, const int&, const cv::Size& );

void add_color( cv::Vec3b& pix, const uchar& b, const uchar& g, const uchar& r );



/*
 * Linear Interpolation
 * p1 - Point 1
 * p2 - Point 2
 * t  - Ratio from Point 1 to Point 2
*/
cv::Point2d lerp( cv::Point2d const& p1, cv::Point2d const& p2, const double& t ){
    return cv::Point2d( ((1-t)*p1.x) + (t*p2.x),
                        ((1-t)*p1.y) + (t*p2.y));
}

/*
 * Interpolate Colors
*/
template <typename DATATYPE, int N>
cv::Vec<DATATYPE,N> lerp( cv::Vec<DATATYPE,N> const& minColor,
                          cv::Vec<DATATYPE,N> const& maxColor,
                          double const& t ){

    cv::Vec<DATATYPE,N> output;
    for( int i=0; i<N; i++ ){
        output[i] = (uchar)(((1-t)*minColor[i]) + (t * maxColor[i]));
    }
    return output;
}

/*
 * Compute the dem color
*/
cv::Vec3b get_dem_color( const double& elevation ){

    // if the elevation is below the minimum, return the minimum
    if( elevation < color_range[0].second ){
        return color_range[0].first;
    }
    // if the elevation is above the maximum, return the maximum
    if( elevation > color_range.back().second ){
        return color_range.back().first;
    }

    // otherwise, find the proper starting index
    int idx=0;
    double t = 0;
    for( int x=0; x<(int)(color_range.size()-1); x++ ){

        // if the current elevation is below the next item, then use the current
        // two colors as our range
        if( elevation < color_range[x+1].second ){
            idx=x;
            t = (color_range[x+1].second - elevation)/
                (color_range[x+1].second - color_range[x].second);

            break;
        }
    }

    // interpolate the color
    return lerp( color_range[idx].first, color_range[idx+1].first, t);
}

/*
 * Given a pixel coordinate and the size of the input image, compute the pixel location
 * on the DEM image.
*/
cv::Point2d world2dem( cv::Point2d const& coordinate, const cv::Size& dem_size   ){


    // relate this to the dem points
    // ASSUMING THAT DEM DATA IS ORTHORECTIFIED
    double demRatioX = ((dem_tr.x - coordinate.x)/(dem_tr.x - dem_bl.x));
    double demRatioY = 1-((dem_tr.y - coordinate.y)/(dem_tr.y - dem_bl.y));

    cv::Point2d output;
    output.x = demRatioX * dem_size.width;
    output.y = demRatioY * dem_size.height;

    return output;
}

/*
 * Convert a pixel coordinate to world coordinates
*/
cv::Point2d pixel2world( const int& x, const int& y, const cv::Size& size ){

    // compute the ratio of the pixel location to its dimension
    double rx = (double)x / size.width;
    double ry = (double)y / size.height;

    // compute LERP of each coordinate
    cv::Point2d rightSide = lerp(tr, br, ry);
    cv::Point2d leftSide  = lerp(tl, bl, ry);

    // compute the actual Lat/Lon coordinate of the interpolated coordinate
    return lerp( leftSide, rightSide, rx );
}

/*
 * Add color to a specific pixel color value
*/
void add_color( cv::Vec3b& pix, const uchar& b, const uchar& g, const uchar& r ){

    if( pix[0] + b < 255 && pix[0] + b >= 0 ){ pix[0] += b; }
    if( pix[1] + g < 255 && pix[1] + g >= 0 ){ pix[1] += g; }
    if( pix[2] + r < 255 && pix[2] + r >= 0 ){ pix[2] += r; }
}


/*
 * Main Function
*/
int main( int argc, char* argv[] ){

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

    // load the image (note that we don't have the projection information.  You will
    // need to load that yourself or use the full GDAL driver.  The values are pre-defined
    // at the top of this file
    //![load1]
    cv::Mat src_limg = (cv::imread(argv[1],0) cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR );
    cv::Mat trg_rimg = (cv::imread(argv[6],0) cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR );
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


// create our output products
cv::Mat output_dem1(src_limg.size(), CV_8UC3 );
cv::Mat output_dem2(trg_rimg.size(), CV_8UC3 );

// for sanity sake, make sure GDAL Loads it as a signed short
if( (dem1.type() or dem2.type())  != CV_16SC1 ){ throw std::runtime_error("DEM image type must be CV_16SC1"); }

// define the color range to create our output DEM heat map
//  Pair format ( Color, elevation );  Push from low to high
//  Note:  This would be perfect for a configuration file, but is here for a working demo.
color_range.push_back( std::pair<cv::Vec3b,double>(cv::Vec3b( 188, 154,  46),   -1));
color_range.push_back( std::pair<cv::Vec3b,double>(cv::Vec3b( 110, 220, 110), 0.25));
color_range.push_back( std::pair<cv::Vec3b,double>(cv::Vec3b( 150, 250, 230),   20));
color_range.push_back( std::pair<cv::Vec3b,double>(cv::Vec3b( 160, 220, 200),   75));
color_range.push_back( std::pair<cv::Vec3b,double>(cv::Vec3b( 220, 190, 170),  100));
color_range.push_back( std::pair<cv::Vec3b,double>(cv::Vec3b( 250, 180, 140),  200));

// define a minimum elevation
double minElevation = -10;

// iterate over each pixel in the source src_limg, computing the dem point
for( int y=0; y<src_limg.rows; y++ ){
    for( int x=0; x<src_limg.cols; x++ ){
    
        // convert the pixel coordinate to lat/lon coordinates
        cv::Point2d coordinate = pixel2world( x, y, src_limg.size() );
    
        // compute the dem src_limg pixel coordinate from lat/lon
        cv::Point2d dem_coordinate = world2dem( coordinate, dem.size() );
    
        // extract the elevation
        double dz;
        if( dem_coordinate.x >=    0    && dem_coordinate.y >=    0     &&
            dem_coordinate.x < dem.cols && dem_coordinate.y < dem.rows ){
            dz = dem.at<short>(dem_coordinate);
        }else{
            dz = minElevation;
        }
    
        // write the pixel value to the file
        output_dem_flood.at<cv::Vec3b>(y,x) = src_limg.at<cv::Vec3b>(y,x);
    
        // compute the color for the heat map output
        cv::Vec3b actualColor = get_dem_color(dz);
        output_dem.at<cv::Vec3b>(y,x) = actualColor;
    
        // show effect of a 10 meter increase in ocean levels
        if( dz < 10 ){
            add_color( output_dem_flood.at<cv::Vec3b>(y,x), 90, 0, 0 );
        }
        // show effect of a 50 meter increase in ocean levels
        else if( dz < 50 ){
            add_color( output_dem_flood.at<cv::Vec3b>(y,x), 0, 90, 0 );
        }
        // show effect of a 100 meter increase in ocean levels
        else if( dz < 100 ){
            add_color( output_dem_flood.at<cv::Vec3b>(y,x), 0, 0, 90 );
        }
    
    }}

// iterate over each pixel in the target trg_rimg, computing the dem point
for( int y=0; y<trg_rimg.rows; y++ ){
    for( int x=0; x<trg_rimg.cols; x++ ){
    
        // convert the pixel coordinate to lat/lon coordinates
        cv::Point2d coordinate = pixel2world( x, y, trg_rimg.size() );
    
        // compute the dem trg_rimg pixel coordinate from lat/lon
        cv::Point2d dem_coordinate = world2dem( coordinate, dem.size() );
    
        // extract the elevation
        double dz;
        if( dem_coordinate.x >=    0    && dem_coordinate.y >=    0     &&
            dem_coordinate.x < dem.cols && dem_coordinate.y < dem.rows ){
            dz = dem.at<short>(dem_coordinate);
        }else{
            dz = minElevation;
        }
    
        // write the pixel value to the file
        output_dem_flood.at<cv::Vec3b>(y,x) = trg_rimg.at<cv::Vec3b>(y,x);
    
        // compute the color for the heat map output
        cv::Vec3b actualColor = get_dem_color(dz);
        output_dem.at<cv::Vec3b>(y,x) = actualColor;
    
        // show effect of a 10 meter increase in ocean levels
        if( dz < 10 ){
            add_color( output_dem_flood.at<cv::Vec3b>(y,x), 90, 0, 0 );
        }
        // show effect of a 50 meter increase in ocean levels
        else if( dz < 50 ){
            add_color( output_dem_flood.at<cv::Vec3b>(y,x), 0, 90, 0 );
        }
        // show effect of a 100 meter increase in ocean levels
        else if( dz < 100 ){
            add_color( output_dem_flood.at<cv::Vec3b>(y,x), 0, 0, 90 );
        }
    
    }}

//Tiling Method @aymenalsaadi :) 
int src_x1= strtol(argv[2],NULL, 10); 
int src_y1= strtol(argv[3],NULL, 10);
int src_x2= strtol(argv[4],NULL, 10);
int src_y2= strtol(argv[5],NULL, 10);
int trg_x1= strtol(argv[7],NULL, 10);
int trg_y1= strtol(argv[8],NULL, 10);
int trg_x2= strtol(argv[9],NULL, 10);
int trg_y2= strtol(argv[10],NULL,10);

cv::Mat limg = cv::Mat(src_limg, cv::Rect(src_x1,src_y1,src_x2,src_y2));
cv::Mat rimg = cv::Mat(trg_rimg, cv::Rect(trg_x1,trg_y1,trg_x2,trg_y2)); 

unsigned int w = limg.cols;
unsigned int h = limg.rows;
unsigned int w2 = rimg.cols;
unsigned int h2 = rimg.rows;

std::string command1 ,command2;
std::string cmd,cmd2="gdalinifo ";
std::cout << "gdal source image 1 inforamtion    " <<system(cmd.c_str())<<std::endl;


std::cout << "Source image size = (" << w << "," << h << ")" << std::endl;
std::cout << "Target image size = (" << w2 << "," << h2 << ")" << std::endl;


std::cout << "Initializing data..." << std::endl;
InitCuda(devNum); 
CudaImage img1, img2;
img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
img1.Download();
img2.Download(); 

// Extract Sift features from images
SiftData siftData1, siftData2;
float initBlur = 0.3f;
//float thresh =3.3f;
float thresh = (imgSet ? 4.5f : 3.0f);
//std::cin>>thresh ; 
//std::cin>>initBlur;
std::cout<<"Threshold value :"<<thresh<<std::endl;
InitSiftData(siftData1, 132768, true, true); //before it was 100,000
InitSiftData(siftData2, 132768, true, true); //before it was 100,000

system("nvidia-smi");
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
  FindHomography(siftData1, homography, &numMatches, 10000, 0.0f, 0.80f, 5.0);
  int numFit = ImproveHomography(siftData1, homography, 5, 0.0f, 0.80f, 3.0);
  
  std::cout << "Number features detected by SIFT descriptors : " <<"Source Image "<<  siftData1.numPts << ", Target Image " << siftData2.numPts << std::endl;
  std::cout << "Number of Matched features: " << numMatches <<std::endl;
  std::cout << "Number of fitted features after applying ImproveHomography : " << numFit <<std::endl;
  std::cout << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " <<std::endl;


string ss1 = (argv[1]);
string ss2 = (argv[6]);

std::cout<<ss1;
char sep = '/';
size_t i = ss1.rfind(sep, ss1.length());
size_t i2 = ss2.rfind(sep, ss2.length());

if (i != string::npos && i2 != string::npos) 
 {
    string filename = ss1.substr(i+1, ss1.length() - i);
    string filename2 = ss2.substr(i2+1, ss2.length() - i2);
    size_t lastindex = filename.find_last_of("."); 
    size_t lastindex2 = filename2.find_last_of(".");
    string rawname = filename.substr(0, lastindex); 
    string rawname2 = filename2.substr(0, lastindex2);
    
    PrintMatchData(siftData1, siftData2, img1, rawname, rawname2);
 }

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

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img, const string& s, const string& s2)
{
  
   //@aymenalsaadi creating a csv file to save the matched keypoints
   std::ofstream myfile;
   string ss1 = s;
   string ss2 = s2;  
   string ff  = string("/pylon5/mc3bggp/aymen/cuda_out/sift_matches_")+ss1+string("_")+ss2+string(".csv");
   const char * c = ff.c_str(); 
   //myfile.open(c);
   myfile.open(c, std::ofstream::out | std::ofstream::trunc);
   myfile<<"x1, y1, sigma1, angle1, t1_x, t1_y, theta1, x2, y2, sigma2, angle2, t2_x, t2_y, theta2"<<std::endl;
   //myfile.close();
     

  //@aymenalsaadi creating a csv file to save the matched keypoints
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
    int x2 = (int)(sift2[j].xpos+0.5);
    int y2 = (int)(sift2[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int s2 = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift2[j].scale)))));
    int p = y*w + x;
    int p2 = y2*w + x2;
    myfile<<(x)<<","<<(y)<<","<<(s)<<","<<(s2)<<","<<(p)<<","<<(p2)<<","<<(x2)<<","<<(y2)<<","<<(s)<<","<<(s2)<<","<<(p)<<","<<(p2)<<","<<(s)<<","<<(p)<<std::endl; 
    
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}
