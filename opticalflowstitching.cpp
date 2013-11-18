#include "opencv2/opencv.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include <iostream>

using namespace cv;
using namespace cv::detail;
const unsigned int NUMBER_OF_FEATURES = 500;

int roundToChosen(int numToRound, int multiple=0){
  if(multiple == 0) 
    return numToRound;
  int remainder = numToRound % multiple;
  if (remainder == 0)
    return numToRound;
  return numToRound - remainder;
} 

Point2f getMostCommonVector(vector<Point2f> vectors, vector<uchar> connectedVectors){
  int biggestCount = 0, vectorCount = 0;
  Point2f mostCommonVector;
  std::map<string, int> vectorCounts;
  for(size_t s=0; s<vectors.size(); s++){
    if(connectedVectors[s]){
      vectorCounts[(int)vectors[s].x+"-"+(int)vectors[s].y] += 1;
      vectorCount = vectorCounts[(int)vectors[s].x+"-"+(int)vectors[s].y];
      if(vectorCount > biggestCount){
        biggestCount = vectorCount;
        mostCommonVector = vectors[s];
      }
    }
  }
  return mostCommonVector;
}

vector<Point2f> minusPoints(vector<Point2f> prevPoints, vector<Point2f> nextPoints){
  vector<Point2f> newVectors;
  for(size_t s=0; s<prevPoints.size(); s++){
    newVectors.push_back(nextPoints[s] - prevPoints[s]);
  }
  return newVectors;
}

Mat differenceBlendJointArea(Mat baseImage, Mat imageToBlend, Point2f vector, Mat baseBackground)
{
  double alpha = 0.5; // how hard the change is blended to image
  Rect imageRect = Rect(0,0,imageToBlend.cols, imageToBlend.rows) & (Rect(0,0,imageToBlend.cols, imageToBlend.rows) + Point(vector.x, vector.y));
  Rect baseRect = Rect(0,0,imageToBlend.cols, imageToBlend.rows) & (Rect(0,0,imageToBlend.cols, imageToBlend.rows) - Point(vector.x, vector.y));

  Mat imageROI(imageToBlend, imageRect);
  Mat baseROI(baseImage, baseRect);

  if(norm(vector)>0.5)
  {
    baseROI.copyTo(imageROI);
    return baseImage;
  }

  for(int x=0;x<baseROI.cols;x++)
    for(int y=0;y<baseROI.rows;y++)
    {
      for(int channel=0; channel<3; channel++)
      {
        int A = baseROI.at<Vec3b>(y,x)[channel];
        int B = imageROI.at<Vec3b>(y,x)[channel];
        int C = baseBackground.at<Vec3b>(y,x)[channel];
        int diff = std::abs(C - B);
        //diff = roundToChosen(diff, 50);
        diff = diff<30?0:diff;
        int diffMult = std::min(1,diff);
        baseROI.at<Vec3b>(y,x)[channel] = (int)(diffMult*alpha*B + diffMult*(1-alpha)*A + (1-diffMult)*A);
        //Different blending methods that can be found from photoshop.
        //Best that is in use is plain alpha blend for diffed image

        //baseROI.at<Vec3b>(y,x)[channel] = ((A < 128) ? (2 * diff * A / 255):(255 - 2 * (255 - diff) * (255 - A) / 255));
        //baseROI.at<Vec3b>(y,x)[channel] = (std::min(255, (A + diff)));
        //baseROI.at<Vec3b>(y,x)[channel] = diff;
        //baseROI.at<Vec3b>(y,x)[channel] = (255 - (((255 - A) * (255 - B)) >> 8));
        //baseROI.at<Vec3b>(y,x)[channel] = (A + B - 2 * A * B / 255);
        //baseROI.at<Vec3b>(y,x)[channel] = ((A == 0) ? A : std::max(0, (255 - ((255 - diff) << 8 ) / A)));
        //baseROI.at<Vec3b>(y,x)[channel] = ((diff == 255) ? B:min(255, ((A << 8 ) / (255 - diff))));
        //baseROI.at<Vec3b>(y,x)[channel] = ((A < 128)?(2*((diff>>1)+64))*((float)A/255):(255-(2*(255-((diff>>1)+64))*(float)(255-A)/255)));
      }
    }
  return baseImage;
}

Mat blendImagesWithDisplacement(Mat baseImage, Mat baseImageMask,
    Mat imageToBlend, Point2f* displacementVector, Mat baseBackground)
{
  //make large enough base image
  Mat outImage(baseImage.rows+abs(displacementVector->y), baseImage.cols+abs(displacementVector->x),CV_8UC3);
  outImage = Scalar(255,255,255); //make the new image black
  Rect inputImageRect = Rect(std::max(displacementVector->x, 0.f), std::max(displacementVector->y, 0.f), imageToBlend.cols, imageToBlend.rows);
  Rect baseImageRect = Rect(abs(std::min(displacementVector->x, 0.f)),abs(std::min(displacementVector->y, 0.f)), baseImage.cols, baseImage.rows);
  Mat blendingRoi(outImage, inputImageRect);
  Mat baseRoi(outImage, baseImageRect);
  imageToBlend.copyTo(blendingRoi);
  baseImage.copyTo(baseRoi);
  //blend
  outImage = differenceBlendJointArea(outImage, imageToBlend, *displacementVector, baseBackground);
  //save the position of latest image
  displacementVector->x = std::max(displacementVector->x, 0.f);
  displacementVector->y = std::max(displacementVector->y, 0.f);
  return outImage;
}

int main(int argc, char* argv[])
{
  if(argc<2)
  {
    std::cout << "This program needs a video for an input!" << std::endl;
    return 0;
  }
  VideoCapture cap(argv[1]);//"/media/PENDRIVE/micname-movies/3shots.mov");
  //VideoCapture cap(0); // open the default camera
  if(!cap.isOpened()){  // check if we succeeded
    std::cout << "failed to load" << std::endl;
    return -1;
  }
  TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  Mat frame, prevFrame, nextFrame;
  vector<uchar> featuresFound;
  vector<Point2f> nextPoints;

  //Initial values
  cap >> frame;
  cvtColor(frame, nextFrame, CV_BGR2GRAY);
  goodFeaturesToTrack(nextFrame, nextPoints, NUMBER_OF_FEATURES, 0.001, 1);
  vector<Point2f> prevPoints(nextPoints);
  nextFrame.copyTo(prevFrame);
  Mat stitchedImage, stitchedImageMask;
  frame.copyTo(stitchedImage);
  stitchedImageMask = Mat::zeros(nextFrame.size(), CV_8UC1);
  Mat baseBackground;
  frame.copyTo(baseBackground);
  Point2f displacementSum(0,0);

  for(int i=0;;i++)
  {
    // new frame
    cap >> frame;
    if(frame.empty())
      break;
    frame.copyTo(nextFrame);
    //Tracking points
    cvtColor(nextFrame, nextFrame, CV_BGR2GRAY);

    // Calculate Optical Flow
    calcOpticalFlowPyrLK(prevFrame, nextFrame, prevPoints, nextPoints, featuresFound, noArray());

    if(i%4==1){
      //get motion vector that is the most common
      Point2f common = getMostCommonVector(minusPoints(prevPoints, nextPoints), featuresFound);
      if(norm(common)>0.5)
      {
        frame.copyTo(baseBackground);
      }
      displacementSum -= common;
      //blend two images to one
      stitchedImage = blendImagesWithDisplacement(stitchedImage, stitchedImageMask, frame, &displacementSum, baseBackground);
      imshow("activeImage", stitchedImage);
      //imshow("activeFrame", frame);
      //imshow("activeBackground", baseBackground);
      //init new tracking points
      goodFeaturesToTrack(nextFrame, nextPoints, NUMBER_OF_FEATURES, 0.001, 1);
      cornerSubPix(nextFrame, nextPoints, Size(10,10), Size(-1,-1), termcrit);
      vector<Point2f> prevPoints(nextPoints);
      nextFrame.copyTo(prevFrame);
    }else{ 
      char key = waitKey(30);
      if(key == 'n')
      {
        string windowName = "last result lasting to" + i;
        imshow(windowName, stitchedImage);
        frame.copyTo(stitchedImage);
        frame.copyTo(baseBackground);
      }else if(key>=0)
        break;
    }
  }
  waitKey();
  return 0;
}
