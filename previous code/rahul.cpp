// Rahul Pillalamarri, Griffin Detracy, Camas Collins 
// CSS 487 Final Project
// The purpose of this implementation file is to create, read, and find a databox in a given real world image. 
// A databox is created by encrypting a string as different color cells such that the box is filled with different color values.
// Borders with a particular ratio as well as a special mark on the top left corner is then created and added to the databox to enable for
// initial identification of the databox.
// The databox is initially found in a real life image by converting the image to greyscale and then iterating through it to find the ratios 
// of the border. Once the ratios are found, [GRIFFIN PLEASE PUT SOME MORE TECH STUFF HERE] the edges are then sent to the opencv hough line function
// in order to detect the inner edges of the databox. If multiple lines for an edge are detected then the average line is found and that is the line that is 
// deemed as the edge. Then we find the intersection points of all the edge lines detected and crop/unskew the image using those 4 interesection/corner points. 
// The reading of the databox is then done by iterating through the image and decrypting the different color values to get the original string. 


#define BIT_ISSET(var, pos) (!!((var) & (1ULL<<(pos))))
#define BORDER_PERCENT_ERROR_THRESHHOLD 40
#define EDGE_THREASHOLD 200
#define BORDER_RATIO 128
#define BORDER_WIDTH_1(size) (size/BORDER_RATIO)
#define _USE_MATH_DEFINES
#define HOUGH_THETA CV_PI/180
#define HOUGH_RESOLUTION 1
#define HOUGH_THRESHOLD(count) (count * .3)
#define HOUGH_THRESHOLD_DECREMENT .02
#define DRAWLINE_MULT 50
#define WHITE_BLACK_THRESH 127
#define HISTOGRAM_SIZE 8
#define BUCKET_SIZE (256/HISTOGRAM_SIZE)
#define STARTER_DATA_OFFSET 8
#define COLOR_DIFF_THRESHOLD 30

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
#include <chrono>

using namespace cv;
using namespace std;

// saving color scalar values to make it easier to call those color values later in the code
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

// pre: a valid Mat Image and Vec2f with two floats that are within the scope of the image are passed into the function
// post: a line is drawn on the image using the Vec2f elements as starting and ending points on the line and the opencv line function
// drawLineStandardXY: converts Vec2f points/elements into valid cartesian coordinates and then graphs a line on the given Mat image
void drawLineStandardXY(Mat& img, Vec2f v) {
    Point pt1(cvRound(v[0] + DRAWLINE_MULT * (-v[1])), cvRound(v[1] + DRAWLINE_MULT * (v[0])));
    Point pt2(cvRound(v[0] - DRAWLINE_MULT * (-v[1])), cvRound(v[1] - DRAWLINE_MULT * (v[0])));
    line(img, pt1, pt2, color[red], 3, LINE_AA);
}

// pre: a valid Mat Image and Vec2f with two floats that are within the scope of the image are passed into the function
// post: a line is drawn on the image using the Vec2f elements as starting and ending points on the line and the opencv line function
// drawLineParametricSpace: converts Vec2f points/elements into valid polar coordinates and then graphs a line on the given Mat image
void drawLineParameterSpace(Mat& img, Vec2f v) {
    double a = cos(v[1]), b = sin(v[1]);
    double x0 = a * v[0], y0 = b * v[0];
    v[0] = x0;
    v[1] = y0;
    return drawLineStandardXY(img, v);
}

// pre: A valid Mat Image and string were passed into the function
// post: The Mat image passed in is then written to the disk under the name of the string passed in
// and is shown to the user immediatly after execution
// showImg: this function is to be able to save and show the image passed into the fucntion to the user. 
// The image passed in is likely an image that has undergone some sort of transformation or processing and the result of such processing is shown
void showImg(const Mat& img, string name, int shrinkVal = 1) {
    namedWindow(name, WINDOW_NORMAL);
    imwrite(name, img);
    resizeWindow(name, Size(img.cols / shrinkVal, img.rows / shrinkVal));
    imshow(name, img);

}

// pre: A valid Mat image (databox) that will be processed, integers to hold the following:
// cellN: number of cells per row in the databox
// cellIndex: index of the cell that needs to be writted with respect to the databox
// val: the bit value that is trying to be encrypted
// offset: where the cell is in the image
// post: A set of pixels called cells are encrypted by color to hold some particular data.
// writeCell: Function encrypts a passed in integer to 
void writeCell(Mat& db, const int cellN, const int cellIndex, const int val, const float offset = 0) {
    float cellWidth = (db.cols - (2 * offset)) / cellN;
    int r = ((cellIndex / cellN) * cellWidth) + offset;
    int c = ((cellIndex % cellN) * cellWidth) + offset;
    for (int row = 0; row < cellWidth; row++) {
        for (int col = 0; col < cellWidth; col++) {
            db.at<Vec3b>(r + row, c + col) = color[val];
        }
    }
}

// pre: a valid Mat Image and a vector of integers that represents the ratios required for the border detection
// post: a Mat image with borders drawn specific to the ratios passed in and with a corner marked for orientation identification is returned 
// drawBorderRatio: The function draws rectangular borders with ratios passed in as the vector ratios. It also marks a corner 
// with a particular pattern which is used later for identifying if an image is oriented the correct way or not.
Mat drawBorderRatio(const Mat& input, vector<int> ratios) {
    const int borderWidth1 = BORDER_WIDTH_1(input.cols);
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
    for (int i = 0; i < offsetFromBorder; i++) {
        for (int j = 0; j < offsetFromBorder; j++) {
            output.at<Vec3b>(i, j) = color[black];
            output.at<Vec3b>(output.cols - 1 - i, j) = color[white];
            output.at<Vec3b>(j, output.cols - 1 - i) = color[white];
            output.at<Vec3b>(output.cols - 1 - j, output.cols - 1 - i) = color[white];
        }
    }

    return output;
}

// pre: a valid Mat image, valid cell number that is within the provided Mat, and a float value is passed into the function
// post: a border is drawn on the Mat image passed into the function
// drawBorderCells: function sets the correct borders with correct ratios for each border for identification
Mat drawBorderCells(const Mat& input, const int cellN, float& borderOffset) {
    Mat output = input.clone();
    bool even = cellN % 2 == 0;
    bool isWhite = 1;

    for (int i = 0; i < cellN; i++) {
        writeCell(output, cellN, i, isWhite, borderOffset);
        writeCell(output, cellN, i + (cellN * (cellN - 1)), (!even == isWhite), borderOffset);
        writeCell(output, cellN, cellN * i, isWhite, borderOffset);
        writeCell(output, cellN, (cellN * i) + (cellN - 1), (!even == isWhite), borderOffset);
        isWhite = !isWhite;
    }

    borderOffset += (float)(output.cols - (2 * borderOffset)) / cellN;
    return output;
}

// pre: ASK GRIFFIN
// post: An databox image is returned 
// makeDataBox: 
Mat makeDataBox(const char* data, const int dataArrSize, const int cellN, const int dbWidth, const int bitsPerCell, const vector<int> ratios) {
    const int dims[] = { dbWidth, dbWidth };
    Mat db(2, dims, CV_8UC3, Scalar::all(0));
    if (bitsPerCell > 3 || bitsPerCell < 1) {
        std::cout << "ERROR: can only do 1,2,3 bits per cell. Input: " << bitsPerCell << endl;
        return db;
    }

    db = drawBorderRatio(db, ratios);
    float borderOffset = 0;
    float borderBase1 = BORDER_WIDTH_1(db.cols);
    for (int i : ratios) borderOffset += (i * borderBase1);

    db = drawBorderCells(db, cellN + 2, borderOffset);
    for (int i = 0; i < STARTER_DATA_OFFSET; i++)
        writeCell(db, cellN, i, i, borderOffset);

    int val = 0;
    int bitIndex = 0;
    while (bitIndex < (dataArrSize * 8)) {
        if ((bitIndex) / bitsPerCell >= (cellN * cellN) - STARTER_DATA_OFFSET) {
            break;
        }
        if (BIT_ISSET(data[bitIndex / 8], bitIndex % 8)) {
            val += pow(2, bitIndex % bitsPerCell);
        }
        if ((bitIndex + 1) % bitsPerCell == 0) {
            writeCell(db, cellN, (((bitIndex + 1) / bitsPerCell) - 1) + STARTER_DATA_OFFSET, val, borderOffset);
            val = 0;
        }
        bitIndex++;
    }
    if (val != 0) writeCell(db, cellN, (((bitIndex + 1) / bitsPerCell) - 1), val, borderOffset);
    return db;
}

// pre: a valid Mat Image and Vec3b pointer is passed in where the Mat image is the image that will be manipulated and the Vec3b pointer
// is for assigning colors to particular cells in the image
// post: the color values of each cell in the patch image is found and that value is returned
// findColorValue: helper function for reading a data box by iterating through the Mat image sent and finding 
// the average color value to determine what the color value of that patch is
char findColorValue(Mat& patch, Vec3b* realColors) {

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
    char val = 0;
    for (int i = 0; i < 8; i++) {
        int temp = 0;
        for (int h = 0; h < 3; h++) {
            temp += abs(realColors[i][h] - average[h]);
        }
        if (temp < closest) {
            closest = temp;
            val = i;
        }
    }
    return val;
}

// pre: two float values are passed in
// post: returns the absolute value of (a-b)*100/a or 0 if a is equal to b
// getError100: this helper function helps find the ratios between the borders by figuring out the error percentage
int getError100(const float a, const float b) {
    if (a - b == 0) return 0; // cant divide by 0, but 0 is perfect val
    return (abs(a - b) * 100) / a;
}

// pre: takes in two vectors of ints, the first one that holds the ratios of the borders found while iterating through the databox
// and the other that holds the true ratios for the border of the databox
// post: returns true if the ratios match and false if they dont
// ratioMatch: function to check the ratio patterns and check to see if they match
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

// pre:
// post:
// ratioMatchFlipped: 
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

// pre: a valid Mat image that has been processed with gaussian blur, greyscale conversion, and canny edge detection is passed in
// as well as a vector that holds valid and true ratios for the borders of the databox for identification
// post: returns a pair of Vec2f elements called output with each element in the pair holding the top and bottom edge lines detected by the function
// findDataBoxWEdgeImageH: function finds the edges/lines that run along the edges of the databox to enable for the detection of the databox in a real world image.
// Does so by initially iterating through the real life image in a top down fashion and finding the ratios of the border for the databox.
// Once the end of the border ratio is found, it then calls the hough line function on those top and bottom edges to accurately identify the lines for those edges
pair<Vec2f, Vec2f> findDataBoxWEdgeImageH(const Mat& input, vector<int> ratios) {
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
                        outputLeft.push_back(Point(lastEdge + currentPattern[6] + currentPattern[7] - 2, r));
                    }
                    else if (ratioMatchFlipped(currentPattern, ratiosFlipped)) {
                        outputRight.push_back(Point(currentPositions[0] - currentPattern[2] - currentPattern[3] + 2, r));
                    }
                }
                lastEdge = c;
            }
        }
        currentPattern.clear();
        currentPositions.clear();
    }
    auto start = std::chrono::high_resolution_clock::now();
    // Inefficient, but make matricies to pass to hough transform API
    Mat outputRightImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));
    for (Point p : outputRight) outputRightImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;
    Mat outputLeftImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));
    for (Point p : outputLeft) outputLeftImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;

    // top horizontal line
    vector<Vec2f> outputLeftLine;
    float thresholdL = HOUGH_THRESHOLD(outputLeft.size());
    while (outputLeftLine.size() == 0) {
        HoughLines(outputLeftImage, outputLeftLine, HOUGH_RESOLUTION, HOUGH_THETA, thresholdL);
        if (outputLeftLine.size() == 0) thresholdL -= outputLeft.size() * HOUGH_THRESHOLD_DECREMENT;
        if (thresholdL <= 0) break;
    }

    // bottom horizontal line
    vector<Vec2f> outputRightLine;
    float thresholdR = HOUGH_THRESHOLD(outputRight.size());
    while (outputRightLine.size() == 0) {
        HoughLines(outputRightImage, outputRightLine, HOUGH_RESOLUTION, HOUGH_THETA, thresholdR);
        if (outputRightLine.size() == 0) thresholdR -= outputLeft.size() * HOUGH_THRESHOLD_DECREMENT;
        if (thresholdR <= 0) break;
    }


    Vec2f lineRight = { 0,0 };
    for (Vec2f v : outputRightLine) {
        double a = cos(v[1]), b = sin(v[1]);
        double x0 = a * v[0], y0 = b * v[0];
        lineRight[0] += x0;
        lineRight[1] += y0;
    }
    if (outputRightLine.size() != 0) {
        lineRight[0] /= outputRightLine.size();
        lineRight[1] /= outputRightLine.size();
    }

    Vec2f lineLeft = { 0,0 };
    for (Vec2f v : outputLeftLine) {
        double a = cos(v[1]), b = sin(v[1]);
        double x0 = a * v[0], y0 = b * v[0];
        lineLeft[0] += x0;
        lineLeft[1] += y0;
    }
    if (outputLeftLine.size() != 0) {
        lineLeft[0] /= outputLeftLine.size();
        lineLeft[1] /= outputLeftLine.size();
    }

    pair<Vec2f, Vec2f> output(lineLeft, lineRight);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time total HOUGH H " << elapsed.count() << " s\n";
    return output;
}

// pre: a valid Mat image that has been processed with gaussian blur, greyscale conversion, and canny edge detection is passed in
// as well as a vector that holds valid and true ratios for the borders of the databox for identification
// post: returns a pair of Vec2f elements called output with each element in the pair holding the left and right edge lines detected by the function
// findDataBoxWEdgeImageV: function finds the edges/lines that run along the edges of the databox to enable for the detection of the databox in a real world image.
// Does so by initially iterating through the real life image in a left to right fashion and finding the ratios of the border for the databox.
// Once the end of the border ratio is found, it then calls the hough line function on the left and right edges to accurately identify the lines for those edges
pair<Vec2f, Vec2f> findDataBoxWEdgeImageV(const Mat& input, vector<int> ratios) {
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
                        outputLeft.push_back(Point(c, lastEdge + currentPattern[6] + currentPattern[7] - 2));
                    }
                    else if (ratioMatchFlipped(currentPattern, ratiosFlipped)) {
                        outputRight.push_back(Point(c, currentPositions[0] - currentPattern[2] - currentPattern[3] + 2));
                    }
                }
                lastEdge = r;
            }
        }
        currentPattern.clear();
        currentPositions.clear();
    }
    auto start = std::chrono::high_resolution_clock::now();
    // Inefficient, but make matricies to pass to hough transform API
    Mat outputRightImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));
    for (Point p : outputRight) outputRightImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;
    Mat outputLeftImage(Size(input.cols, input.cols), CV_8UC1, Scalar(0));
    for (Point p : outputLeft) outputLeftImage.at<Vec<uchar, 1>>(p.y, p.x)[0] = 255;

    vector<Vec2f> outputLeftLine;
    float thresholdL = HOUGH_THRESHOLD(outputLeft.size());
    while (outputLeftLine.size() == 0) {
        HoughLines(outputLeftImage, outputLeftLine, HOUGH_RESOLUTION, HOUGH_THETA, thresholdL);
        if (outputLeftLine.size() == 0) thresholdL -= outputLeft.size() * HOUGH_THRESHOLD_DECREMENT;
        if (thresholdL <= 0) break;
    }

    // bottom horizontal line
    vector<Vec2f> outputRightLine;
    float thresholdR = HOUGH_THRESHOLD(outputRight.size());
    while (outputRightLine.size() == 0) {
        HoughLines(outputRightImage, outputRightLine, HOUGH_RESOLUTION, HOUGH_THETA, thresholdR);
        if (outputRightLine.size() == 0) thresholdR -= outputLeft.size() * HOUGH_THRESHOLD_DECREMENT;
        if (thresholdR <= 0) break;
    }

    Vec2f lineRight = { 0,0 };
    for (Vec2f v : outputRightLine) {
        double a = cos(v[1]), b = sin(v[1]);
        double x0 = a * v[0], y0 = b * v[0];
        lineRight[0] += x0;
        lineRight[1] += y0;
    }
    if (outputRightLine.size() != 0) {
        lineRight[0] /= outputRightLine.size();
        lineRight[1] /= outputRightLine.size();
    }

    Vec2f lineLeft = { 0,0 };
    for (Vec2f v : outputLeftLine) {
        double a = cos(v[1]), b = sin(v[1]);
        double x0 = a * v[0], y0 = b * v[0];
        lineLeft[0] += x0;
        lineLeft[1] += y0;
    }
    if (outputLeftLine.size() != 0) {
        lineLeft[0] /= outputLeftLine.size();
        lineLeft[1] /= outputLeftLine.size();
    }

    pair<Vec2f, Vec2f> output(lineLeft, lineRight);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time total HOUGH V " << elapsed.count() << " s\n";
    return output;
}

// pre: two Vec2f objects that hold two lines
// post: returns a point that either holds the max float values to indicate that there is no intersection between both lines
// or the point of intersection of two lines passed into the function is returned
// findWallIntersection: functions uses the standard form ax+by=c to find the intersection point of two given lines
Point findWallIntersection(Vec2f v1, Vec2f v2) {
    // NOT QUITE SURE WHAT THIS IS
    Point l1s(v1[0] - v1[1], v1[0] + v1[1]);
    Point l1e(v1[0] + v1[1], -v1[0] + v1[1]);
    Point l2s(v2[0] - v2[1], v2[0] + v2[1]);
    Point l2e(v2[0] + v2[1], -v2[0] + v2[1]);

    // finds the standard form of line 1
    double a1 = l1e.y - l1s.y;
    double b1 = l1s.x - l1e.x;
    double c1 = a1 * (l1s.x) + b1 * (l1s.y);

    // finds the standard form of line 2
    double a2 = l2e.y - l2s.y;
    double b2 = l2s.x - l2e.x;
    double c2 = a2 * (l2s.x) + b2 * (l2s.y);
    double determinant = a1 * b2 - a2 * b1;

    if (determinant == 0)
    {
        // no intersection point, lines are parallel and therefore a nonsensical point is returned
        return Point(FLT_MAX, FLT_MAX);
    }
    else
    {
        // intersection point exists and is calculated and returned
        double x = (b2 * c1 - b1 * c2) / determinant;
        double y = (a1 * c2 - a2 * c1) / determinant;
        return Point(x, y);
    }
}

// pre: two Vec2f objects that hold two lines
// post: returns a point that either holds the max float values to indicate that there is no intersection between both lines
// or the point of intersection of two lines passed into the function is returned
// findCellIntersection: functions uses the standard form ax+by=c to find the intersection point of two given lines
Point findCellIntersection(Point startingPointA, Point endingPointA, Point startingPointB, Point endingPointB) {

    // finds the standard form of line 1
    double a1 = endingPointA.y - startingPointA.y;
    double b1 = startingPointA.x - endingPointA.x;
    double c1 = a1 * (startingPointA.x) + b1 * (startingPointA.y);

    // finds the standard form of line 2
    double a2 = endingPointB.y - startingPointB.y;
    double b2 = startingPointB.x - endingPointB.x;
    double c2 = a2 * (startingPointB.x) + b2 * (startingPointB.y);
    double determinant = a1 * b2 - a2 * b1;

    if (determinant == 0)
    {
        // no intersection point, lines are parallel and therefore a nonsensical point is returned
        return Point(FLT_MAX, FLT_MAX);
    }
    else
    {
        // intersection point exists and is calculated and returned
        double x = (b2 * c1 - b1 * c2) / determinant;
        double y = (a1 * c2 - a2 * c1) / determinant;
        return Point(x, y);
    }
}

// pre: a valid mat image, as well as valid corner points for the Mat image are passed into the function
// post: returns the degree of rotation neceassary to put image right side up by using the special pattern on the top left part of the databox. If image is already 
// rightside up then the function returns -1
// findRotation: function uses the top left corner pattern to determine whether or not the image needs to be rotated and if it does then by how much
int findRotation(const Mat& img, Point2f* srcPoints, Point2f* dstPointsForRotation) {
    Mat colorCheck;
    Mat Matrix = getPerspectiveTransform(srcPoints, dstPointsForRotation);
    warpPerspective(img, colorCheck, Matrix, Size(500, 500));
    if (colorCheck.at<Vec3b>(0, 0)[0] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(0, 0)[1] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(0, 0)[2] < WHITE_BLACK_THRESH)
        return -1;
    if (colorCheck.at<Vec3b>(0, colorCheck.cols - 1)[0] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(0, colorCheck.cols - 1)[1] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(0, colorCheck.cols - 1)[2] < WHITE_BLACK_THRESH)
        return ROTATE_90_COUNTERCLOCKWISE;
    if (colorCheck.at<Vec3b>(colorCheck.rows - 1, 0)[0] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(colorCheck.rows - 1, 0)[1] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(colorCheck.rows - 1, 0)[2] < WHITE_BLACK_THRESH)
        return ROTATE_90_CLOCKWISE;
    if (colorCheck.at<Vec3b>(colorCheck.rows - 1, colorCheck.cols - 1)[0] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(colorCheck.rows - 1, colorCheck.cols - 1)[1] < WHITE_BLACK_THRESH && colorCheck.at<Vec3b>(colorCheck.rows - 1, colorCheck.cols - 1)[2] < WHITE_BLACK_THRESH)
        return ROTATE_180;
    return -1;
}

// pre: a valid Mat image of the daabox in a real life image, and a vector of ratios that correctly corrospond to the ratios shown in the border of the databox are passed in
// post: the databox in the real world image is found, cropped, unskewed, properly oriented, and returned from the function
// findDataBox: function finds the databox in a multi-step process:
//                  Preprocessing: gaussian blur, greyscale conversion, and canny edge detection is done on the image
//                  The image is sent to the findDataBoxWEdgeH and findDataBoxWEdgeV functions to identify the lines/edges of the databox through houghlines and the border ratios
//                  The image is then converted back to rgb and the lines detected from the previous step is graphed. Then helper functions are used to find the points of 
//                  intersection between all the lines such that the corners could be used to unskew the image using the rotation, warpPerspective, and getPerspective functions
//                  The unskewed image is then returned.
Mat findDataBox(const Mat& img, vector<int> ratios) {
    Mat edgeImage = img.clone();
    Mat unskewed_image = img.clone();
    auto start = std::chrono::high_resolution_clock::now();
    GaussianBlur(edgeImage, edgeImage, Size(7, 7), 2, 2);
    cvtColor(edgeImage, edgeImage, COLOR_BGR2GRAY);
    Canny(edgeImage, edgeImage, 20, 60);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time total edge detect: " << elapsed.count() << " s\n";
    start = std::chrono::high_resolution_clock::now();
    pair<Vec2f, Vec2f> pointsH = findDataBoxWEdgeImageH(edgeImage, ratios);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time total find H: " << elapsed.count() << " s\n";
    start = std::chrono::high_resolution_clock::now();
    pair<Vec2f, Vec2f> pointsV = findDataBoxWEdgeImageV(edgeImage, ratios);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time total find V: " << elapsed.count() << " s\n";

    bool broke = pointsV.first[0] == 0 && pointsV.first[1] == 0 && pointsV.second[0] == 0 && pointsV.second[1] == 0;
    if (broke) {
        Mat rot = getRotationMatrix2D(Point(edgeImage.cols / 2, edgeImage.rows / 2), 45, 1.0);
        warpAffine(edgeImage, edgeImage, rot, Size(edgeImage.rows, edgeImage.cols));
        warpAffine(unskewed_image, unskewed_image, rot, Size(edgeImage.rows, edgeImage.cols));

        pointsH = findDataBoxWEdgeImageH(edgeImage, ratios);
        pointsV = findDataBoxWEdgeImageV(edgeImage, ratios);
    }
    broke = pointsV.first[0] == 0 && pointsV.first[1] == 0 && pointsV.second[0] == 0 && pointsV.second[1] == 0;
    if (broke) {
        cout << "ERROR, could not find data box even after rotating 45 degrees" << endl << endl;
        return img;
    }
    Mat EdgeCopy = edgeImage.clone();
    GaussianBlur(EdgeCopy, EdgeCopy, Size(5, 5), 2, 2);

    cvtColor(EdgeCopy, EdgeCopy, COLOR_GRAY2BGR);
    drawLineStandardXY(EdgeCopy, pointsH.first);
    drawLineStandardXY(EdgeCopy, pointsH.second);
    drawLineStandardXY(EdgeCopy, pointsV.first);
    drawLineStandardXY(EdgeCopy, pointsV.second);


    Point p = findWallIntersection(pointsH.first, pointsV.first);
    Point2f c1 = findWallIntersection(pointsH.first, pointsV.first);
    Point2f c2 = findWallIntersection(pointsH.first, pointsV.second);
    Point2f c3 = findWallIntersection(pointsH.second, pointsV.first);
    Point2f c4 = findWallIntersection(pointsH.second, pointsV.second);

    // unskew the image
    Mat rotationImg = unskewed_image.clone();
    Point2f srcPoints[] = { c1, c3, c2, c4 };
    Point2f dstPoints[] = { Point(0,0), Point(500, 0), Point(0, 500), Point(500, 500) };
    Point2f dstPointsForRotation[] = { Point(40,40), Point(460, 40), Point(40, 460), Point(460, 460) };
    Mat Matrix = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(unskewed_image, unskewed_image, Matrix, Size(500, 500));

    int rotation = findRotation(rotationImg, srcPoints, dstPointsForRotation);
    if (rotation != -1) {
        rotate(unskewed_image, unskewed_image, rotation);
    }

    waitKey(0);
    return unskewed_image;
}

// pre: two scalars, one that holds the true value for a color and the other that holds the value that needs to be compared, and an integer for the thresholod is passed in
// post: if the colors have a greater difference than the threshold value specified then return false, otherwise return true
// colorMatchVec3b: Function is used to compare two color values to check to see if a particular color is the color that we believe it is
bool colorMatchVec3b(const Scalar colorTrue, const Scalar colorCheck, const int threshError) {
    for (int i = 0; i < 3; i++) {
        if (abs(colorTrue[i] - colorCheck[i]) > threshError) return false;
    }
    return true;
}

// pre: a valid Mat image that corrosponds to a particular cell in the databox is passed into the function
// post: The average color value of that cell is found and returned 
// getAvgColorVal: helper function to find the average color value of a cell to make it easier to read the databox 
Vec3b getAvgColorVal(Mat& cell) {
    int aveB = 0;
    int aveG = 0;
    int aveR = 0;
    int pixelCount = cell.rows * cell.cols;
    for (int row = 0; row < cell.rows; row++) {
        for (int col = 0; col < cell.cols; col++) {
            aveB += cell.at<Vec3b>(row, col)[0];
            aveG += cell.at<Vec3b>(row, col)[1];
            aveR += cell.at<Vec3b>(row, col)[2];
        }
    }
    aveB = aveB / pixelCount;
    aveG = aveG / pixelCount;
    aveR = aveR / pixelCount;
    Vec3b c;
    c[0] = aveB;
    c[1] = aveG;
    c[2] = aveR;
    return c;
}

// pre: passes in the databox, matrix that holds the center points of the cells, and the number of bits per cell
// post: returns the decrypted string that the databox was holding
// readDataBox: function reads the databox by using the findCellsPoints function to get the matrix that 
// holds the center coordinates of each cell in each element of the matrix. It then iterates through the matrix 
// to get the color of the cell at each coordinate. The color is then converted to a bit that is associated with each char.
// By doing so the function incrementally converts each cell into a char to eventually build out the encrypted string
string readDataBox(const Mat& db, const Mat& cellCenters, const int bitsPerCell) {
    Vec3b curColors[8];
    int cellWidth = cellCenters.at<Vec<int, 2>>(1, 0)[1] - cellCenters.at<Vec<int, 2>>(0, 0)[1];
    int cellHeight = cellCenters.at<Vec<int, 2>>(0, 1)[0] - cellCenters.at<Vec<int, 2>>(0, 0)[0];
    cellWidth = cellWidth / 4;
    cellHeight = cellHeight / 4;
    String output = "";

    vector<bool> currentChar = { 0,0,0,0,0,0,0,0 };
    int currentCharIndex = 0;
    for (int cRow = 0; cRow < cellCenters.rows; cRow++) {
        for (int cCol = 0; cCol < cellCenters.cols; cCol++) {
            int r = cellCenters.at<Vec<int, 2>>(cRow, cCol)[0] - cellHeight;
            int c = cellCenters.at<Vec<int, 2>>(cRow, cCol)[1] - cellWidth;
            Mat cell;
            try {
                cell = db(Rect(r, c, 2 * cellHeight, 2 * cellWidth));
            }
            catch (exception e) { return output; }
            if (cCol < 8 && cRow == 0) {
                curColors[cCol] = getAvgColorVal(cell);
            }
            else {
                char colorPos = findColorValue(cell, curColors);
                for (int i = 0; i < bitsPerCell; i++) {
                    if (BIT_ISSET(colorPos, i)) {
                        currentChar[currentCharIndex] = 1;
                    }
                    currentCharIndex++;
                    if (currentCharIndex == currentChar.size()) {
                        currentCharIndex = 0;
                        char c = 0;
                        for (int j = 0; j < currentChar.size(); j++) {
                            if (currentChar[j])
                                c += pow(2, j);
                        }
                        currentChar = { 0,0,0,0,0,0,0,0 };
                        output += (char)c;
                    }
                }
            }
        }
    }
    return output;
}

// pre: takes in 3 integers, first two for x and y coordinates and the third for the size of the cell
// post: returns a point that is on the corrosponding walls
// pointTopWallImg, pointBottomWallImg, pointLeftWallImg, pointRightWallImg: 
// map functions to map to each wall of a cell based on a given a and b 
Point pointTopWallImg(const int a, const int b, const int size)
{
    return Point(a, b);
}

Point pointBottomWallImg(const int a, const int b, const int size)
{
    return Point(size - 1, b);
}

Point pointLeftWallImg(const int a, const int b, const int size)
{
    return Point(b, a);
}

Point pointRightWallImg(const int a, const int b, const int size)
{
    return Point(b, size - 1);
}

// pre: takes in a valid Mat image (databox) and the number of cells in a row in the databox is passed into the function
// post: returns a matrix where each element in the matrix holds the coordinates for the centers of each cell in the databox passed in
// findCellPoints: iterating through the Mat input (databox) black and white borders to find the clear defined edges.
// this is then used to iterate through an edge cell to find the center of the cell. This is done at every cell
// in order to dynamically figure what the average center of each cell is going to be. This is then used to get 
// the most accurate color value for that cell.
Mat findCellPoints(const Mat& input, const int cellN) {
    int cellNBorder = cellN + 2;
    const int dimsOWB[] = { cellNBorder, cellNBorder };
    Mat outputWBorder(2, dimsOWB, CV_32SC2, Scalar::all(0)); // each cell hold (row,col) of its pixel
    Mat edgeImage = input.clone();
    //Mat DrawPoints = input.clone();
    float cellWidth = edgeImage.cols / cellNBorder;

    int cd2 = cellWidth / 2; // cell divided by 2
    Scalar blackRef = input.at<Vec3b>(cd2, cd2); // white pixel taken from image
    Scalar whiteRef = input.at<Vec3b>(cd2, input.cols - cd2); // black pixel taken from image
    Point(*getPointImg[4])(const int, const int, const int) = { pointTopWallImg, pointBottomWallImg, pointLeftWallImg, pointRightWallImg };

    int lastBlack = 0;
    int lastWhite = 0;
    int lastEdge = 0;
    int countWall = 0;
    int countNothing = 0;

    Mat inputMod = input.clone();

    GaussianBlur(inputMod, inputMod, Size(7, 7), 2, 2);
    Mat DrawPoints = inputMod.clone();

    for (int j = 0; j < 4; j++) {
        for (int i = cd2; i < edgeImage.cols; i++) {
            bool isWhite = colorMatchVec3b(inputMod.at<Vec3b>(getPointImg[j](cd2, i, input.cols - cd2)), whiteRef, COLOR_DIFF_THRESHOLD);
            bool isBlack = colorMatchVec3b(inputMod.at<Vec3b>(getPointImg[j](cd2, i, input.cols - cd2)), blackRef, COLOR_DIFF_THRESHOLD);

            if (isWhite) {
                if (lastBlack > lastWhite) { // most recent was black AKA color change Edge detected
                    int avg = (i + lastEdge - countNothing - 1) / 2;
                    Point p = getPointImg[j](cd2, avg, input.cols - cd2);
                    outputWBorder.at<Vec<int, 2>>(getPointImg[j](0, countWall, cellNBorder)) = { p.x, p.y }; // row, col format
                    circle(DrawPoints, p, 3, color[red], 2);
                    showImg(DrawPoints, "DrawPoints.jpg");
                    waitKey(0);
                    countWall++;
                    if (countWall == cellNBorder) break;
                    lastEdge = i;
                    i += cd2;
                }
                countNothing = 0;
                lastWhite = i;
            }
            else if (isBlack) {
                if (lastBlack < lastWhite) { // most recent was black AKA color change Edge detected
                    int avg = (i + lastEdge - countNothing - 1) / 2;
                    Point p = getPointImg[j](cd2, avg, input.cols - cd2);
                    outputWBorder.at<Vec<int, 2>>(getPointImg[j](0, countWall, cellNBorder)) = { p.x, p.y }; // row, col format
                    circle(DrawPoints, p, 3, color[red], 2);
                    showImg(DrawPoints, "DrawPoints.jpg");
                    waitKey(0);
                    countWall++;
                    if (countWall == cellNBorder) break;
                    lastEdge = i;
                    i += cd2;
                }
                countNothing = 0;
                lastBlack = i;
            }
            else countNothing++;
        }
        lastBlack = 0;
        lastWhite = 0;
        lastEdge = 0;
        countWall = 0;
        countNothing = 0;
    }

    const int dims[] = { cellN, cellN };
    Mat output(2, dims, CV_32SC2, Scalar::all(0)); // each cell hold (row,col) of its pixel

    for (int r = 1; r < cellNBorder - 1; r++) {
        for (int c = 1; c < cellNBorder - 1; c++) {
            Point prs = Point(outputWBorder.at<Vec<int, 2>>(r, 0)[0], outputWBorder.at<Vec<int, 2>>(r, 0)[1]);
            Point pre = Point(outputWBorder.at<Vec<int, 2>>(r, cellNBorder - 1)[0], outputWBorder.at<Vec<int, 2>>(r, cellNBorder - 1)[1]);
            Point pcs = Point(outputWBorder.at<Vec<int, 2>>(0, c)[0], outputWBorder.at<Vec<int, 2>>(0, c)[1]);
            Point pce = Point(outputWBorder.at<Vec<int, 2>>(cellNBorder - 1, c)[0], outputWBorder.at<Vec<int, 2>>(cellNBorder - 1, c)[1]);
            Point p = findCellIntersection(prs, pre, pcs, pce);
            circle(DrawPoints, p, 3, color[red], 2);
            showImg(DrawPoints, "DrawPoints.jpg");
            waitKey(0);
            output.at<Vec<int, 2>>(r - 1, c - 1) = p;
        }
    }
    return output;
}

int main(int argc, char* argv[])
{
    string s = "Computer vision is the study of methods to extract content from digital images/video for applications such as recognizing people, navigating in an unfamiliar environment, image-based rendering, and retrieving images from a vast library. This class will examine all stages of computer vision from low-level to high-level. Low-level tasks include noise reduction and edge detection. In the middle, are estimating 3D properties of the scene, determining the camera motion from an image sequence, and segmentation of an image into coherent subsets. At the highest level, object ///recognition and scene interpretation are performed.";
    std::cout << "length = " << s.length() << endl;

    // set data for use
    const char* sd = s.data();
    char* dataOut = (char*)malloc(sizeof(char) * (s.length() + 1));
    vector<int> borderRatios = { 2, 6, 2, 3, 2, 2, 1, 1, 3, 2 }; // ratio 2:6:2:3:2:2:1:1:3:2 total p = 24 = 6 cells (DO NOT CHANGE) RatioMatch function has HARD CODE
    int cellN = 24;
    int resolution = 1024;
    int bitsPerCell = 3;


    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    Mat overlay = makeDataBox(sd, s.length(), cellN, resolution, bitsPerCell, borderRatios);
    // Record start time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    showImg(overlay, "DataBox.jpg");
    waitKey(0);

    Mat real = imread("real1.jpg");

    start = std::chrono::high_resolution_clock::now();
    Mat result = findDataBox(real, borderRatios);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time total find: " << elapsed.count() << " s\n";

    showImg(result, "result.jpg");
    waitKey(0);

    Mat unskewed = imread("result.jpg");
    Mat cellPoints = findCellPoints(unskewed, cellN);

    string fromDB = readDataBox(unskewed, cellPoints, bitsPerCell);

    cout << " output From DB: " << fromDB << endl;

    return 0;
}
