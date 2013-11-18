#include "opencv2/opencv.hpp"
#include <iostream>
#include <map>
#include <string.h>

using namespace cv;

const unsigned int NUMBER_OF_FEATURES = 500;

int roundToChosen(int numToRound){
  return numToRound;
  int multiple = 2;
   if(multiple == 0) 
     return numToRound;
   int remainder = numToRound % multiple;
   if (remainder == 0)
    return numToRound;
   return numToRound + multiple - remainder;
} 

Point2f getMostCommonVector(vector<Point2f> vectors, vector<uchar> connectedVectors){
  int biggestCount = 0, vectorCount = 0;
  Point2f biggestVector;
  std::map<string, int> vectorCounts;
  for(size_t s=0; s<vectors.size(); s++){
    if(connectedVectors[s]){
      vectorCounts[roundToChosen(vectors[s].x)+"-"+roundToChosen(vectors[s].y)] += 1;
      vectorCount = vectorCounts[roundToChosen(vectors[s].x)+"-"+roundToChosen(vectors[s].y)];
      if(vectorCount > biggestCount){
        biggestCount = vectorCount;
        biggestVector = vectors[s];
      }
    }
  }
  return biggestVector;
}

vector<Point2f> subtractPoints(vector<Point2f> prevPoints, vector<Point2f> nextPoints){
  vector<Point2f> newVectors;
  for(size_t s=0; s<prevPoints.size(); s++){
    newVectors.push_back(nextPoints[s] - prevPoints[s]);
  }
  return newVectors;
}

double calculateMovementTotal(vector<Point2f> movementVector)
{
  double sum = 0;
  for(unsigned int i=0; i<movementVector.size(); i++)
    sum += norm(movementVector[i]);
  return sum;
}

vector<double> updateMovementDifference(vector<Point2f> movementVector, vector<double> differenceVector, double* lastSum)
{
  double movementTotal = calculateMovementTotal(movementVector);
  if(differenceVector.size()==0)
  {
    *lastSum = movementTotal;
    differenceVector.push_back((double)0);
    return differenceVector;
  }
  double difference = std::abs(*lastSum - movementTotal);
  differenceVector.push_back(difference);
  return differenceVector;
}

void drawGraphToImage(Mat image, vector<double> dataPoints, Scalar color)
{
  Point2f lastPoint(0,image.rows), newPoint;
  double stepSize = (double)image.cols / dataPoints.size();
  double maxVal = *std::max_element(dataPoints.begin(), dataPoints.end());
  for(unsigned int i=0; i<dataPoints.size();i++)
  {
    newPoint = Point(i*stepSize,(1-(double)dataPoints[i]/maxVal)*image.rows);
    line(image, lastPoint, newPoint, color, 2);
    lastPoint = newPoint;
  }
}

Mat updateGraph(vector<double> movementDifference, vector<double> histogramDifference)
{
  Mat updatedGraph(510,1024,CV_8UC3);
  updatedGraph = Scalar(255,255,255);
  drawGraphToImage(updatedGraph, movementDifference, Scalar(255,0,255));
//  drawGraphToImage(updatedGraph, histogramDifference, Scalar(255,0,0));
  return updatedGraph;
}

MatND calculateHistogram(Mat image)
{
  MatND hist;
  Mat hsv_base;
  //image to HSV
  cvtColor( image, hsv_base, CV_BGR2HSV );

  /// Using 30 bins for hue and 32 for saturation
  int h_bins = 50; int s_bins = 60;
  int histSize[] = { h_bins, s_bins };

  // hue varies from 0 to 256, saturation from 0 to 180
  float h_ranges[] = { 0, 256 };
  float s_ranges[] = { 0, 180 };

  const float* ranges[] = { h_ranges, s_ranges };

  // Use the o-th and 1-st channels
  int channels[] = { 0, 1 };

  //histogram calculation
  calcHist( &hsv_base, 1, channels, Mat(), hist, 2, histSize, ranges, true, false );
  normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );

  return hist;
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
  int loopWaitTime = 30;
  TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  Mat frame, prevFrame, nextFrame;
  vector<uchar> featuresFound;
  vector<Point2f> nextPoints;

  //variables for statistics
  vector<double> movementDifference;
  vector<double> histogramDifference;

  MatND lastHistogram;
  double lastVectorSum = 0;

  //Initial values
  cap >> frame;
  frame.copyTo(nextFrame);
  cvtColor(nextFrame, nextFrame, CV_BGR2GRAY);
  goodFeaturesToTrack(nextFrame, nextPoints, 500, 0.001, 1);
  vector<Point2f> prevPoints(nextPoints);
  Point2f mostCommonVector = Point2f(10,10);
  nextFrame.copyTo(prevFrame);
  lastHistogram = calculateHistogram(frame);

  for(int i=0;;i++)
  {
    // new frame
    cap >> frame;
    if(frame.empty())
      break;
    frame.copyTo(nextFrame);
    cvtColor(nextFrame, nextFrame, CV_BGR2GRAY);

    // Calculate Optical Flow
    calcOpticalFlowPyrLK(prevFrame, nextFrame, prevPoints, nextPoints, featuresFound, noArray());
    imshow("optical flow", frame);

    if(i%10==1)
    {
      //calculate movement vectors and histogram
      vector<Point2f> movementVectors = subtractPoints(prevPoints, nextPoints);
      MatND hist = calculateHistogram(frame);

      //get statistics out of movement and hist
      movementDifference = updateMovementDifference(movementVectors, movementDifference, &lastVectorSum);
      histogramDifference.push_back(compareHist( hist, lastHistogram, 1 ));
      lastHistogram = hist;

      Mat graph = updateGraph(movementDifference, histogramDifference);
      imshow("graph", graph);

      mostCommonVector = getMostCommonVector(movementVectors, featuresFound);
      goodFeaturesToTrack(nextFrame, nextPoints, 500, 0.001, 1);
      cornerSubPix(nextFrame, nextPoints, Size(10,10), Size(-1,-1), termcrit);
      vector<Point2f> prevPoints(nextPoints);
      nextFrame.copyTo(prevFrame);
    } else {
      char key = waitKey(loopWaitTime);
      if(key == ' ')
      {
        waitKey();
      }else if(key>=0){
        break;
      }
    }
  }
  std::cout << "Everything went better than expected!" << std::endl;
  waitKey();
  return 0;
}
