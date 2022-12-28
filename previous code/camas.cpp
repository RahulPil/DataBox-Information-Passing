#define BIT_ISSET(var, pos) (!!((var) & (1ULL<<(pos))))
#define BORDER_PERCENT_ERROR_THRESHHOLD 40
#define EDGE_THREASHOLD 200
#define BORDER_CELL_RATIO 4
#define _USE_MATH_DEFINES
#define HOUGH_THETA CV_PI/180
#define HOUGH_RESOLUTION 1
#define HOUGH_THRESHOLD(count) (count * .3)
#define HOUGH_THRESHOLD_DECREMENT .02
#define DRAWLINE_MULT 50
#define WHITE_BLACK_THRESH 127
#define HISTOGRAM_SIZE 8
#define BUCKET_SIZE (256/HISTOGRAM_SIZE)
#define STARTER_DATA_OFFSET 1

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

void drawLineStandardXY(Mat& img, Vec2f v) {
    Point pt1(cvRound(v[0] + DRAWLINE_MULT * (-v[1])), cvRound(v[1] + DRAWLINE_MULT * (v[0])));
    Point pt2(cvRound(v[0] - DRAWLINE_MULT * (-v[1])), cvRound(v[1] - DRAWLINE_MULT * (v[0])));
    line(img, pt1, pt2, color[white], 1, LINE_AA);
}

void drawLineParameterSpace(Mat& img, Vec2f v) {
    double a = cos(v[1]), b = sin(v[1]);
    double x0 = a * v[0], y0 = b * v[0];
    v[0] = x0;
    v[1] = y0;
    return drawLineStandardXY(img, v);
}

void showImg(const Mat& img, string name, int shrinkVal = 1) {
    namedWindow(name, WINDOW_NORMAL);
    imwrite(name, img);
    resizeWindow(name, Size(img.cols / shrinkVal, img.rows / shrinkVal));
    imshow(name, img);

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

    writeCell(db, cellN, 0, white, borderOffset);

    int val = 0;
    int bitIndex = 0;
    while (bitIndex < (dataArrSize * 8) - STARTER_DATA_OFFSET) {
        if ((bitIndex) / bitsPerCell >= (cellN * cellN)) {
            //std::cout << "ERROR: to much data for this box" << endl;
            //std::cout << "ERROR: missed " << (dataArrSize * 8) - bitIndex << " bits or " << ((dataArrSize * 8) - bitIndex) / 8 << " bytes" << endl;
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

string toAscii(string binaryValues) {
    stringstream sstream(binaryValues);
    string text = "";
    while (sstream.good()) {
        bitset<8> bits;
        sstream >> bits;
        char c = char(bits.to_ulong());
        text += c;
    }
    return text;
}

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

Vec3b getAveColorVal(Mat& cell) {
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

void RDBm2(const Mat& db, const Mat& cellCenters) {
    Vec3b curColors[8];
    int bitsPerCell = 3;
    int cellWidth = cellCenters.at<Vec<uchar, 2>>(0, 1)[1] - cellCenters.at<Vec<uchar, 2>>(0, 0)[1];
    int cellHeight = cellCenters.at<Vec<uchar, 2>>(1, 0)[0] - cellCenters.at<Vec<uchar, 2>>(0, 0)[0];
    cellWidth = cellWidth / 4;
    cellHeight = cellHeight / 4;
    char curLet = NULL;
    String binaryString = "";
    int bitPosition = 0;


    for (int cRow = 0; cRow < cellCenters.rows; cRow++) {
        for (int cCol = 0; cCol < cellCenters.cols; cCol++) {
            int r = cellCenters.at<Vec<uchar, 2>>(cRow, cCol)[0] - cellHeight;
            int c = cellCenters.at<Vec<uchar, 2>>(cRow, cCol)[0] - cellWidth;
            Mat cell;
            try {
                cell = db(Rect(r, c, 2 * cellHeight, 2 * cellWidth));
            }
            catch (exception e) { return; }
            if (cCol < 8 && cRow == 0) {
                curColors[cCol] = getAveColorVal(cell);
            }
            else {
                char colorPos = findColorValue(cell, curColors);
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
    }
    string outputMessage = toAscii(binaryString);
    std::cout << "output: " << outputMessage << endl;
    ofstream message("OutputMessage.txt");
    message << outputMessage;
    message.close();
}

void readDataBox(const Mat& db, const int bitsPerCell, const int resolution) {
    //
    //    if (bitsPerCell > 3 || bitsPerCell < 1) {
    //        std::cout << "ERROR: can only do 1,2,3 bits per cell. Input: " << bitsPerCell << endl;
    //        return;
    //    }
    cout << endl << "binary string: ";
    int cellWidth = (db.cols) / (resolution - 1);
    int cellHeight = (db.rows) / (resolution - 1);
    int dims[2] = { db.rows, db.cols };
    Mat output(2, dims, CV_8UC3);
    string binaryString = "";
    int bitPosition = 0;
    char curLet = NULL;
    Vec3b realColors[8];
    int count = 0;
    for (int yCellPos = (cellHeight / 2); yCellPos < db.rows; yCellPos += cellHeight) {
        for (int xCellPos = (cellWidth / 2); xCellPos < db.cols; xCellPos += cellWidth) {

            int r = yCellPos - (cellHeight / 4);
            int c = xCellPos - (cellWidth / 4);
            int w = (cellWidth / 4) * 2;
            int h = (cellHeight / 4) * 2;
            Mat patch;
            try {
                patch = db(Rect(r, c, w, h));
            }
            catch (Exception e) {
                return;
            }

            for (int cr = 0; cr < patch.rows; cr++) {
                for (int cc = 0; cc < patch.cols; cc++) {
                    output.at<Vec3b>(c + cc, r + cr) = patch.at<Vec3b>(cc, cr);
                }
            }

            if (count < 8) {
                realColors[count] = getAveColorVal(patch);
            }
            else {
                char colorPos = findColorValue(patch, realColors);
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
            count++;
        }
    }
    string outputMessage = toAscii(binaryString);
    std::cout << endl << "output: " << outputMessage << endl;
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

Point findWallIntersection(Vec2f v1, Vec2f v2) {
    Point l1s(v1[0] - v1[1], v1[0] + v1[1]);
    Point l1e(v1[0] + v1[1], -v1[0] + v1[1]);
    Point l2s(v2[0] - v2[1], v2[0] + v2[1]);
    Point l2e(v2[0] + v2[1], -v2[0] + v2[1]);

    double a1 = l1e.y - l1s.y;
    double b1 = l1s.x - l1e.x;
    double c1 = a1 * (l1s.x) + b1 * (l1s.y);

    // Line CD represented as a2x + b2y = c2
    double a2 = l2e.y - l2s.y;
    double b2 = l2s.x - l2e.x;
    double c2 = a2 * (l2s.x) + b2 * (l2s.y);

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
        //showImg(edgeImage, "Rotated by 45 Degrees.jpg");
        //waitKey(0);
        pointsH = findDataBoxWEdgeImageH(edgeImage, ratios);
        pointsV = findDataBoxWEdgeImageV(edgeImage, ratios);
    }
    broke = pointsV.first[0] == 0 && pointsV.first[1] == 0 && pointsV.second[0] == 0 && pointsV.second[1] == 0;
    if (broke) {
        cout << "ERROR, could not find data box even after rotating 45 degrees" << endl << endl;
        return img;
    }
    //Mat EdgeCopy = edgeImage.clone();
    //GaussianBlur(EdgeCopy, EdgeCopy, Size(5, 5), 2, 2);
    //showImg(EdgeCopy, "edgeImageBefore.jpg", 4);
    ////waitKey(0);
    //cvtColor(EdgeCopy, EdgeCopy, COLOR_GRAY2BGR);
    //drawLineStandardXY(EdgeCopy, pointsH.first);
    //drawLineStandardXY(EdgeCopy, pointsH.second);
    //drawLineStandardXY(EdgeCopy, pointsV.first);
    //drawLineStandardXY(EdgeCopy, pointsV.second);
    //showImg(EdgeCopy, "edgeImageAfter.jpg", 4);
    ////waitKey(0);

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

    //showImg(unskewed_image, "unskew.jpg");
    waitKey(0);
    return unskewed_image;
}

int main(int argc, char* argv[])
{
    //string s = "da";
    //string s = "test data: our names are griffin, camas, and rahul. we are the coolest";
    //string s = "test data: our names are griffin, camas, and rahul. we are the coolest. we are in the css487 class with professor olsen. rahul's aaaaaaaa";
    string s = "Computer vision is the study of methods to extract content from digital images/video for applications such as recognizing people, navigating in an unfamiliar environment, image-based rendering, and retrieving images from a vast library. This class will examine all stages of computer vision from low-level to high-level. Low-level tasks include noise reduction and edge detection. In the middle, are estimating 3D properties of the scene, determining the camera motion from an image sequence, and segmentation of an image into coherent subsets. At the highest level, object ///recognition and scene interpretation are performed.";
    //string s = "Goals of Project: Build and save a DataBox containing an encoded string as an array of colored cells. From a image containing a Data Box->find, crop, and transform the DataBox to a readable DataBox. Analyze the color pattern in the found DataBoxand return the original string. (Under Image: Built DataBox using 3-bits per color, data from this slide)";
    //string s = "Accomplishments: Creation of a new data passing system. Developed a ratio identification system based on ratio space from an edge image.Advantage is Speed: only parsing Image once Horizontally and Vertically O(2n^2), Scale is irrelevant because of ratio matching. Disadvantage is Edge Dependent : bad Canny on borders will result in failure or bad detecting. Implementation of data box object detection through the use of ratio identificationand Hough Transform (Under Image: Built data box of this slide)";
    //string s = "What We Learned: Deeper understanding of pattern recognition in a still image. Knowledge about the struggles color recognition in inconsistent brightnesses and potential fixes. Strengthened our understanding of mathematical integration into CV image identification. Understanding how to detect lines and corners based off of real input images that contain the databox in the general image. (Under Image: Built data box of this slide)";
    //string s = "Understanding the Data Box (Non CV): Border of the image has a specific ratio between edges for detection. Ratio Shown: 2 : 6 : 2 : 3 : 2 : 2 : 1 : 1 : 3 : 2. Black Corner represents Top Left and helps the detector rotate the image correctly for data retrieval. Data is displayed as octal.Each color represents 3 bits(0 - 7) black = 000(0) | white = 001(1) | red = 010(2) green = 011(3) | yellow = 100(4) | blue = 101(5) magenta = 110(6) | cyan = 111(7). Data Box has 32x32 cells which can hold a total of 384 bytes";
    //string s = "Steps For Project: Create Data Box. Encrypt input string to bits and  converted to color values based on a cypher where:. each color represents 3 bits based on its position in the array of colors. Create an identification border with modifiable ratios of white to black bordersIdentify Data Box(CV).Takes in a picture from a camera of a scene containing a Data Box and perform edge detection.Iterate through the edge image to find instances of the identification border ration between edges.Identify the edges of the DataBox using the Hough spaceCropand unskew the Data Box to a readable square.Read the Data Box(CV).Iterate through the cells in the DataBoxand adds their associated binary bit value to an array.Translates the binary string to ASCII";
    //std::cout << "length = " << s.length() << endl;
    //const char* sd = s.data();
    //char* dataOut = (char*)malloc(sizeof(char) * (s.length() + 1));


    //vector<int> borderRatios = { 2, 6, 2, 3, 2, 2, 1, 1, 3, 2 }; // ratio 2:6:2:3:2:2:1:1:3:2 total p = 24 = 6 cells (DO NOT CHANGE) RatioMatch function has HARD CODE

    // Record start time
    //auto start = std::chrono::high_resolution_clock::now();
    //Mat overlay = makeDataBox(sd, s.length(), 32, 1024, 3, borderRatios);
    // Record start time
    //auto finish = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    //GaussianBlur(overlay, overlay, Size(7, 7), 2, 2);
    //cvtColor(overlay, overlay, COLOR_BGR2GRAY);
    //Canny(overlay, overlay, 20, 60);
    //cvtColor(overlay, overlay, COLOR_GRAY2BGR);
    //GaussianBlur(overlay, overlay, Size(7, 7), 2, 2);
    //showImg(overlay, "overlayEdgeDetected.jpg");
    //waitKey(0);

    Mat real = imread("test6.jpg");
    readDataBox(real, 3, 24);

    //start = std::chrono::high_resolution_clock::now();
    //Mat result = findDataBox(real, borderRatios);
    //finish = std::chrono::high_resolution_clock::now();
    //elapsed = finish - start;
    //std::cout << "Elapsed time total find: " << elapsed.count() << " s\n";
    //readDataBox(result,3, 32); 
    //showImg(result, "result.jpg");
    //waitKey(0);

    return 0;
}
