#ifndef _code
#define _code

#include <vector>

typedef std::vector< std::vector<double> > Mat;
typedef std::vector<double> Vec;

const int KERNAL_DX = 0;
const int KERNAL_DY = 1;
const int KERNAL_DXY = 2;

struct Keypoint {
    Keypoint( int xx, int yy, double s ) : x( xx ), y( yy ), scale( s ) {}
    double scale;
    int x;
    int y;
    Vec des;
};

Mat imintegral(Mat im);
double fastsum(Mat imint, int l, int t, int w, int h);
double fashmask(Mat imint, int left, int top, int size, int kernal);
double matmin(Mat mat);
double matmax(Mat mat);
void normalize(Mat &mat);
Mat convolute(Mat imint, int size=9, int kernal=KERNAL_DX);
int scale2size(double scale);
double size2scale(int size);
Mat discriminant(Mat imint, int size, double w=0.9);
std::vector<Keypoint> nonmaxima(std::vector<Mat> octave, double th = 0.6);
std::vector<Keypoint> detect(Mat im, int octave=3, int period=4, double scale=1.2);
void extract(vector<Keypoint> keypoints);
#endif
