/**
 *  \brief Automatic brightness and contrast optimization with optional histogram clipping
 *  \param [in]src Input image GRAY or BGR or BGRA
 *  \param [out]dst Destination image 
 *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
 *  \note In case of BGRA image, we won't touch the transparency
**/
// Default Headers and libraries
// Ref : http://www.programmersought.com/article/9935110603/
#include <iostream>
#include <cmath>
#include <iomanip>

// OpenCV Headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

/*An opencv utiliy */
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0)
{

    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }
}

int main(int argc, char* argv[] )
{
    cv::Mat src_limg = (cv::imread('../source_image.tif'))
    cv::Mat adap_src_limg
    cv::Mat trg_rimg = (cv::imread('../target_image.tif'))
    cv::Mat adap_trg_rimg
    BrightnessAndContrastAuto(src_limg,adap_src_limg,1)
    BrightnessAndContrastAuto(trg_rimg,adap_trg_rimg,1)

    

}
