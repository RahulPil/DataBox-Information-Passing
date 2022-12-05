#define BIT_ISSET(var, pos) (!!((var) & (1ULL<<(pos))))
#define BORDER_PERCENT_ERROR_THRESHHOLD 30
#define EDGE_THREASHOLD 200
#define BORDER_CELL_RATIO 4
#define _USE_MATH_DEFINES

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <bitset>
#include <cmath>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <opencv2/features2d.hpp>
#include <cmath>

using namespace cv;
using namespace std;

const Vec3b color[8] = {
   Vec3b(0, 0, 0),       // black   - 000
   Vec3b(255, 255, 255), // white   - 001
   Vec3b(0, 0, 255),     // red     - 010
   Vec3b(0, 255, 0),     // green   - 011
   Vec3b(0, 255, 255),   // yellow  - 100
   Vec3b(255, 0, 0),     // blue    - 101   
   Vec3b(255, 0, 255),   // magenta - 110
   Vec3b(255, 255, 0)    // cyan    - 111
};
const enum {
    black = 0,
    white,
    red,
    green,
    yellow,
    blue,
    magenta,
    cyan
};

void showImg(const Mat& img, string name) {
    namedWindow(name);
    imshow(name, img);
    cv::imwrite(name, img);
}

void writeCell(Mat& db, const int cellN, const int cellIndex, const int val, const int offset = 0) {
    int cellWidth = (db.cols - (2 * offset)) / cellN;
    int r = ((cellIndex / cellN) * cellWidth) + offset;
    int c = ((cellIndex % cellN) * cellWidth) + offset;
    for (int row = 0; row < cellWidth; row++) {
        for (int col = 0; col < cellWidth; col++) {
            db.at<Vec3b>(r + row, c + col) = color[val];
        }
    }
}

Mat drawBorder(const Mat& input, vector<int> ratios, const int cellN) {
    const int borderWidth1 = input.cols / cellN / BORDER_CELL_RATIO; // border width for 1 ratio
    Mat output = input.clone();

    // draw all border
    int currentRatio = 0;
    int offsetFromBorder = 0;
    bool isWhite = true;
    while (currentRatio < ratios.size()) {
        for (int i = offsetFromBorder; i < (ratios[currentRatio] * borderWidth1) + offsetFromBorder; i++) {
            for (int j = offsetFromBorder; j < input.cols - offsetFromBorder; j++) {
                output.at<Vec3b>(j, i) = color[isWhite];
                output.at<Vec3b>(i, j) = color[isWhite];
                output.at<Vec3b>(j, input.cols - 1 - i) = color[isWhite];
                output.at<Vec3b>(input.cols - 1 - i, j) = color[isWhite];
            }
        }
        offsetFromBorder += (ratios[currentRatio] * borderWidth1); // set offset to point left at
        currentRatio++;
        isWhite = !isWhite;
    }

    // add cover top left corner with black and all others with white
    Mat cat = imread("cat.jpg");
    resize(cat, cat, Size(offsetFromBorder, offsetFromBorder));
    for (int i = 0; i < offsetFromBorder; i++) {
        for (int j = 0; j < offsetFromBorder; j++) {
            output.at<Vec3b>(i, j) = cat.at<Vec3b>(i, j);
        }
    }

    //// draw triange poining to corner
    //for (int i = offsetFromBorder / 5; i < offsetFromBorder - (offsetFromBorder / 5); i++) {
    //   for (int j = offsetFromBorder / 5; j < offsetFromBorder - (offsetFromBorder / 5); j++) {
    //      if (i + j < offsetFromBorder)
    //         output.at<Vec3b>(i, j) = color[white];
    //   }
    //}

    return output;
}

Mat makeDataBox(const char* data, const int dataArrSize, const int cellN, const int dbWidth, const int bitsPerCell, const vector<int> ratios) {
    const int dims[] = { dbWidth, dbWidth };
    const int cellWidth = dbWidth / cellN;
    Mat db(2, dims, CV_8UC3, Scalar::all(0));
    if (bitsPerCell > 3 || bitsPerCell < 1) {
        std::cout << "ERROR: can only do 1,2,3 bits per cell. Input: " << bitsPerCell << endl;
        return db;
    }

    db = drawBorder(db, ratios, cellN);
    int borderOffset = 0;
    int BorderBase1 = db.cols / cellN / BORDER_CELL_RATIO;
    for (int i : ratios) borderOffset += (i * BorderBase1);

    int val = 0;
    int bitIndex = 0;
    while (bitIndex < (dataArrSize * 8)) {
        if ((bitIndex) / bitsPerCell >= (cellN * cellN)) {
            //std::cout << "ERROR: to much data for this box" << endl;
            //std::cout << "ERROR: missed " << (dataArrSize * 8) - bitIndex << " bits or " << ((dataArrSize * 8) - bitIndex) / 8 << " bytes" << endl;
            break;
        }
        if (BIT_ISSET(data[bitIndex / 8], bitIndex % 8)) {
            //std::cout << "1";
            val += pow(2, bitIndex % bitsPerCell);


        }
        else cout << "0";
        if ((bitIndex + 1) % bitsPerCell == 0) {
            writeCell(db, cellN, (((bitIndex + 1) / bitsPerCell) - 1), val, borderOffset);
            val = 0;
        }
        bitIndex++;
    }
    if (val != 0) writeCell(db, cellN, (((bitIndex + 1) / bitsPerCell) - 1), val, borderOffset);
    return db;
}


Mat sharpenDB(const Mat& db) {

    Mat output = db.clone();

    Mat_<Vec3b>::iterator it = output.begin<Vec3b>(), itEnd = output.end<Vec3b>();
    for (; it != itEnd; ++it) {

        (*it)[0] = ((*it)[0] < 128) ? 0 : 255;
        (*it)[1] = ((*it)[1] < 128) ? 0 : 255;
        (*it)[1] = ((*it)[1] < 128) ? 0 : 255;
    }
    cv::imwrite("sharpenedUnskewedDB.jpg", output);
    return output;
}

//int getResolution(const Mat& db) {
//
//}

string toAscii(string binaryValues) {
    stringstream sstream(binaryValues);
    string text;
    while (sstream.good()) {
        bitset<8> bits;
        sstream >> bits;
        char c = char(bits.to_ulong());
        text += c;
    }
    return text;
}

char findColorValue(Mat& patch) {
    int average[] = { 0, 0, 0 };
    int size = patch.rows * patch.cols;

    for (int r = 0; r < patch.rows; r++) {
        for (int c = 0; c < patch.cols; c++) {
            for (int a = 0; a < 3; a++) {
                average[a] += patch.at<Vec3b>(r, c)[a];
            }
        }
    }

    for (int j = 0; j < 3; j++) {
        average[j] = average[j] / size;
    }

    int closest = 255;
    char val = NULL;
    for (int i = 0; i < 8; i++) {
        int temp = 0;
        for (int h = 0; h < 3; h++) {
            temp += abs(color[i][h] - average[h]);
        }
        if (temp < closest) {
            closest = temp;
            val = i;
        }
    }
    return val;
}

void readDataBox(const Mat& inputDB, const int bitsPerCell, const int resolution, int lastRatio) {

    if (bitsPerCell > 3 || bitsPerCell < 1) {
        std::cout << "ERROR: can only do 1,2,3 bits per cell. Input: " << bitsPerCell << endl;
        return;
    }

    Mat db = sharpenDB(inputDB);

    int borderWidth = lastRatio * (db.cols / resolution / BORDER_CELL_RATIO);

    Mat db2 = db(Rect(borderWidth, borderWidth, db.cols - 2 * borderWidth, db.rows - 2 * borderWidth));
    imwrite("CheckNoBorders.jpg", db2);

    const int bitCount = bitsPerCell * resolution * resolution;
    int cellWidth = (db.cols - (2 * borderWidth)) / resolution;
    int cellHeight = (db.rows - (2 * borderWidth)) / resolution;
    string binaryString = "";
    int bitPosition = 0;
    char curLet = NULL;
    for (int xCellPos = (cellWidth / 2) + borderWidth; xCellPos < db.cols; xCellPos += cellWidth) {
        for (int yCellPos = (cellHeight / 2) + borderWidth; yCellPos < db.rows; yCellPos += cellHeight) {


            int r = yCellPos - (cellHeight / 3);
            int c = xCellPos - (cellWidth / 3);
            int w = (cellWidth / 3) * 2;
            int h = (cellHeight / 3) * 2;

            Mat patch = db(Rect(r, c, w, h));
            char colorPos = findColorValue(patch);

            for (int l = 0; l < bitsPerCell; l++) {
                if (bitPosition < 8) {
                    if (BIT_ISSET(colorPos, l)) {
                        curLet += pow(2, bitPosition);
                        std::cout << 1;
                    }
                    else {
                        std::cout << 0;
                    }
                    bitPosition++;
                }
                else {
                    bitPosition = 0;
                    binaryString += curLet;
                    curLet = NULL;
                    if (BIT_ISSET(colorPos, l)) {
                        curLet += pow(2, bitPosition);
                        std::cout << 1;
                    }
                    else {
                        std::cout << 0;
                    }
                    bitPosition++;
                }
            }

        }
    }

    std::cout << endl << "binary string output: " << binaryString << endl;
    string outputMessage = toAscii(binaryString);
    std::cout << "output: " << outputMessage << endl;
    ofstream message("OutputMessage.txt");
    message << outputMessage;
    message.close();
}

int getError100(const float a, const float b) {
    if (a - b == 0) return 0; // cant divide by 0, but 0 is perfect val
    return (abs(a - b) * 100) / a;
}

bool ratioMatch(vector<int> ratioFound, vector<int> ratioDB) {
    for (int i = 1; i < ratioDB.size() - 1; i++) {
        float avgBase = (ratioFound[6] + ratioFound[7]) / 2; // HARD CODED ()
        float val = ratioFound[i] / avgBase; // divide by the first number for ratio
        if (getError100((float)ratioDB[i], val) > BORDER_PERCENT_ERROR_THRESHHOLD) {
            return false;
        }
    }
    return true;
}

bool ratioMatchFlipped(vector<int> ratioFound, vector<int> ratioDB) {
    for (int i = 1; i < ratioDB.size() - 1; i++) {
        float avgBase = (ratioFound[2] + ratioFound[3]) / 2; // HARD CODED ()
        float val = ratioFound[i] / avgBase; // divide by the first number for ratio
        if (getError100((float)ratioDB[i], val) > BORDER_PERCENT_ERROR_THRESHHOLD) {
            return false;
        }
    }
    return true;
}

pair<vector<Vec2f>, vector<Vec2f>> findDataBoxWEdgeImageH(const Mat& input, vector<int> ratios, int& pointCount) {
    vector<Point> outputLeft;
    vector<Point> outputRight;
    vector<int> currentPattern;
    vector<int> currentPositions;
    vector<int> ratiosFlipped;
    for (int i = ratios.size() - 1; i >= 0; i--)
        ratiosFlipped.push_back(ratios[i]);

    for (int r = 0; r < input.rows; r++) {
        int lastEdge = -1;
        for (int c = 0; c < input.cols; c++) {
            bool edge = input.at<Vec<uchar, 1>>(r, c)[0] > EDGE_THREASHOLD;
            if (edge) {
                currentPattern.push_back(c - lastEdge);
                currentPositions.push_back(c);
                if (currentPattern.size() > ratios.size()) {
                    for (int i = 0; i < currentPattern.size() - 1; i++) {
                        currentPattern[i] = currentPattern[i + 1];
                        currentPositions[i] = currentPositions[i + 1];
                    }
                    currentPattern.pop_back();
                    currentPositions.pop_back();
                }
                if (currentPattern.size() == ratios.size()) {
                    if (ratioMatch(currentPattern, ratios)) {
                        outputLeft.push_back(Point(lastEdge, r));
                        //circle(output, Point(lastEdge, r), 3, color[red]);
                        //cout << r << " " << c << " found the edge bro" << endl;
                    }
                    else if (ratioMatchFlipped(currentPattern, ratiosFlipped)) {
                        outputRight.push_back(Point(currentPositions[0], r));
                        //circle(output, Point(currentPositions[0], r), 3, color[green]);
                        //cout << r << " " << c << " found the edge bro" << endl;
                    }
                }
                lastEdge = c;
            }
        }
        currentPattern.clear();
        currentPositions.clear();
    }

    //vector<vector<Point>> output = { outputLeft, outputRight };
    Mat outputRightImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));
    Mat outputLeftImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));

    for (Point p : outputLeft) outputLeftImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;
    for (Point p : outputRight) outputRightImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;

    vector<Vec2f> outputRightLine;
    vector<Vec2f> outputLeftLine;
    pointCount = outputLeft.size();

    // left vertical line super confusing
    HoughLines(outputLeftImage, outputLeftLine, 1, CV_PI / 180, pointCount * 0.15, 0, 0); // runs the actual detection


    // right vertical line super confusing
    pointCount = outputRight.size();
    HoughLines(outputRightImage, outputRightLine, 1, CV_PI / 180, pointCount * 0.15, 0, 0); // runs the actual detection


    //showImg(output, "output of points.jpg");
    //waitKey(0);
    pair<vector<Vec2f>, vector<Vec2f>> output(outputLeftLine, outputRightLine);
    return output;
}

pair<vector<Vec2f>, vector<Vec2f>> findDataBoxWEdgeImageV(const Mat& input, vector<int> ratios, int& pointCount) {
    vector<Point> outputLeft;
    vector<Point> outputRight;
    vector<int> currentPattern;
    vector<int> currentPositions;
    vector<int> ratiosFlipped;
    for (int i = ratios.size() - 1; i >= 0; i--)
        ratiosFlipped.push_back(ratios[i]);

    for (int c = 0; c < input.cols; c++) {
        int lastEdge = -1;
        for (int r = 0; r < input.rows; r++) {
            bool edge = input.at<Vec<uchar, 1>>(r, c)[0] > EDGE_THREASHOLD;
            if (edge) {
                currentPattern.push_back(r - lastEdge);
                currentPositions.push_back(r);
                if (currentPattern.size() > ratios.size()) {
                    for (int i = 0; i < currentPattern.size() - 1; i++) {
                        currentPattern[i] = currentPattern[i + 1];
                        currentPositions[i] = currentPositions[i + 1];
                    }
                    currentPattern.pop_back();
                    currentPositions.pop_back();
                }
                if (currentPattern.size() == ratios.size()) {
                    if (ratioMatch(currentPattern, ratios)) {
                        outputLeft.push_back(Point(c, lastEdge));
                        //circle(output, Point(lastEdge, r), 3, color[red]);
                        //cout << r << " " << c << " found the edge bro" << endl;
                    }
                    else if (ratioMatchFlipped(currentPattern, ratiosFlipped)) {
                        outputRight.push_back(Point(c, currentPositions[0]));
                        //circle(output, Point(currentPositions[0], r), 3, color[green]);
                        //cout << r << " " << c << " found the edge bro" << endl;
                    }
                }
                lastEdge = r;
            }
        }
        currentPattern.clear();
        currentPositions.clear();
    }

    //vector<vector<Point>> output = { outputLeft, outputRight };
    Mat outputRightImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));
    Mat outputLeftImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));

    for (Point p : outputLeft) outputLeftImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;
    for (Point p : outputRight) outputRightImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;

    vector<Vec2f> outputRightLine;
    vector<Vec2f> outputLeftLine;
    pointCount = outputLeft.size();

    // top horizontal line
    HoughLines(outputLeftImage, outputLeftLine, 1, CV_PI / 180, pointCount * 0.30, 0, 0); // runs the actual detection

    // bottom horizontal line
    pointCount = outputRight.size();
    HoughLines(outputRightImage, outputRightLine, 1, CV_PI / 180, pointCount * 0.15, 0, 0); // runs the actual detection


    //showImg(output, "output of points.jpg");
    //waitKey(0);
    pair<vector<Vec2f>, vector<Vec2f>> output(outputLeftLine, outputRightLine);
    return output;
}

Point find_intersection(Point startingPointA, Point endingPointA,
    Point startingPointB, Point endingPointB) {

    double a1 = endingPointA.y - startingPointA.y;
    double b1 = startingPointA.x - endingPointA.x;
    double c1 = a1 * (startingPointA.x) + b1 * (startingPointA.y);

    // Line CD represented as a2x + b2y = c2
    double a2 = endingPointB.y - startingPointB.y;
    double b2 = startingPointB.x - endingPointB.x;
    double c2 = a2 * (startingPointB.x) + b2 * (startingPointB.y);

    double determinant = a1 * b2 - a2 * b1;

    if (determinant == 0)
    {
        // The lines are parallel. This is simplified
        // by returning a pair of FLT_MAX
        return Point(FLT_MAX, FLT_MAX);
    }
    else
    {
        double x = (b2 * c1 - b1 * c2) / determinant;
        double y = (a1 * c2 - a2 * c1) / determinant;
        return Point(x, y);
    }
}

Mat findDataBox(const Mat& img, vector<int> ratios) {
    Mat edgeImage = img.clone();
    Mat linesOut = img.clone();
    Mat output = img.clone();

    GaussianBlur(edgeImage, edgeImage, Size(7, 7), 2, 2);
    cvtColor(edgeImage, edgeImage, COLOR_BGR2GRAY);
    Canny(edgeImage, edgeImage, 20, 60);
    int pointsCount = 0;

    pair<vector<Vec2f>, vector<Vec2f>> pointsH = findDataBoxWEdgeImageH(edgeImage, ratios, pointsCount);

    vector<Vec2f> horizontal_left_line = pointsH.first;
    vector<Vec2f> horizontal_right_line = pointsH.second;

    pair<vector<Vec2f>, vector<Vec2f>> pointsV = findDataBoxWEdgeImageV(edgeImage, ratios, pointsCount);

    vector<Vec2f> vertical_left_line = pointsV.first;
    vector<Vec2f> vertical_right_line = pointsV.second;

    cvtColor(edgeImage, edgeImage, COLOR_GRAY2BGR);

    Point pt_h_top1, pt_h_top2;
    for (size_t i = 0; i < horizontal_left_line.size(); i++)
    {
        float rho = horizontal_left_line[i][0], theta = horizontal_left_line[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 4000 * (-b));
        pt1.y = cvRound(y0 + 4000 * (a));
        pt2.x = cvRound(x0 - 4000 * (-b));
        pt2.y = cvRound(y0 - 4000 * (a));
        pt_h_top1.x = pt1.x;
        pt_h_top1.y = pt1.y;
        pt_h_top2.x = pt2.x;
        pt_h_top2.y = pt2.y;
        line(edgeImage, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    Point pt_h_bottom1, pt_h_bottom2;
    for (size_t i = 0; i < horizontal_right_line.size(); i++)
    {
        float rho = horizontal_right_line[i][0], theta = horizontal_right_line[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 4000 * (-b));
        pt1.y = cvRound(y0 + 4000 * (a));
        pt2.x = cvRound(x0 - 4000 * (-b));
        pt2.y = cvRound(y0 - 4000 * (a));
        pt_h_bottom1.x = pt1.x;
        pt_h_bottom1.y = pt1.y;
        pt_h_bottom2.x = pt2.x;
        pt_h_bottom2.y = pt2.y;
        line(edgeImage, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    Point pt_v_left1, pt_v_left2;
    for (size_t i = 0; i < vertical_left_line.size(); i++)
    {
        float rho = vertical_left_line[i][0], theta = vertical_left_line[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 4000 * (-b));
        pt1.y = cvRound(y0 + 4000 * (a));
        pt2.x = cvRound(x0 - 4000 * (-b));
        pt2.y = cvRound(y0 - 4000 * (a));
        pt_v_left1.x = pt1.x;
        pt_v_left1.y = pt1.y;
        pt_v_left2.x = pt2.x;
        pt_v_left2.y = pt2.y;
        line(edgeImage, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    Point pt_v_right1, pt_v_right2;
    for (size_t i = 0; i < vertical_right_line.size(); i++)
    {
        float rho = vertical_right_line[i][0], theta = vertical_right_line[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 4000 * (-b));
        pt1.y = cvRound(y0 + 4000 * (a));
        pt2.x = cvRound(x0 - 4000 * (-b));
        pt2.y = cvRound(y0 - 4000 * (a));
        pt_v_right1.x = pt1.x;
        pt_v_right1.y = pt1.y;
        pt_v_right2.x = pt2.x;
        pt_v_right2.y = pt2.y;
        line(edgeImage, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }


    // intersection for horizontal top
    Point corner1 = find_intersection(pt_h_top1, pt_h_top2, pt_v_left1, pt_v_left2);
    Point corner2 = find_intersection(pt_h_top1, pt_h_top2, pt_v_right1, pt_v_right2);
    Point corner3 = find_intersection(pt_h_bottom1, pt_h_bottom2, pt_v_left1, pt_v_left2);
    Point corner4 = find_intersection(pt_h_bottom1, pt_h_bottom2, pt_v_right1, pt_v_right2);

    // highlights the corners to check to ensure that we got the right corners, you are gonna have to zoom in a bit tho
    circle(edgeImage, corner1, 3, color[green]);
    circle(edgeImage, corner2, 3, color[green]);
    circle(edgeImage, corner3, 3, color[green]);
    circle(edgeImage, corner4, 3, color[green]);

    Mat unskewed_image;

    // unskew the image
    Point2f srcPoints_for_perspective[] = {
       corner1, corner3, corner2, corner4
    };

    Point2f dstPoints_for_perspective[] = {
        Point(0,0), Point(500, 0), Point(0, 500), Point(500, 500)
    };

    Mat Matrix = getPerspectiveTransform(srcPoints_for_perspective, dstPoints_for_perspective);
    warpPerspective(img, unskewed_image, Matrix, Size(500, 500));
    //showImg(unskewed_image, "unskew.jpg");

    //showImg(edgeImage, "test.jpg");
    //waitKey(0);
    return output;
}

int main(int argc, char* argv[])
{
    string s = "da";
    //string s = "test data: our names are griffin, camas, and rahul. we are the coolest";
    //string s = "test data: our names are griffin, camas, and rahul. we are the coolest. we are in the css487 class with professor olsen. rahul's aaaaaaaa";
    //string s = "Computer vision is the study of methods to extract content from digital images/video for applications such as recognizing people, navigating in an unfamiliar environment, image-based rendering, and retrieving images from a vast library. This class will examine all stages of computer vision from low-level to high-level. Low-level tasks include noise reduction and edge detection. In the middle, are estimating 3D properties of the scene, determining the camera motion from an image sequence, and segmentation of an image into coherent subsets. At the highest level, object ///recognition and scene interpretation are performed.";
    std::cout << "length = " << s.length() << endl;
    const char* sd = s.data();
    char* dataOut = (char*)malloc(sizeof(char) * (s.length() + 1));

    vector<int> borderRatios = { 2, 6, 2, 3, 2, 2, 1, 1, 3, 2 }; // ratio 1:6:2:3:2:2 total p = 16 = 4 cells (DO NOT CHANGE) RatioMatch function has HARD CODE

    Mat overlay = makeDataBox(sd, s.length(), 32, 1024, 3, borderRatios);
    //showImg(overlay, "ovalaaaa.jpg");
    //waitKey(0);
    //readDataBox(overlay, 1, 32);

    Mat overlay2 = overlay.clone();

    Mat real = imread("real.jpg");
    //showImg(real, "real.jpg");

    //namedWindow("overlay2 Image", WINDOW_NORMAL);
    //resizeWindow("overlay2 Image", overlay2.cols / 2, overlay2.rows / 2);
    //imshow("overlay2 Image", overlay2);
    cv::imwrite("overlay2.jpg", overlay2);
    overlay = findDataBox(real, borderRatios);


    //namedWindow("overlay Image", WINDOW_NORMAL);
    //resizeWindow("overlay Image", overlay.cols / 2, overlay.rows / 2);
    //imshow("overlay Image", overlay);
    cv::imwrite("overlay.jpg", overlay);

    Mat foundDB = imread("unskew.jpg");

    readDataBox(foundDB, 3, 32, 2);

    waitKey(0);



    return 0;
}
