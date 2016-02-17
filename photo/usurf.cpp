#include <vector>
#include <cmath>
#include "usurf.h"

using namespace std;

Mat createMat(int h, int w, double value) {
    Mat mat;
    mat.resize(h);
    for (int i=0;i<h;i++) {
        mat[i].resize(w);
        for (int j=0;j<w;j++) {
            mat[i][j] = value;
        }
    }

    return mat;
}


Mat imintegral(Mat im) {
    int h = im.size();
    int w = im[0].size();

    Mat res = createMat(h, w, 0);

    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            if (i==0 && j==0) {
                res[i][j] = im[i][j];
            }
            else if (i==0) {
                res[i][j] = im[i][j] + res[i][j-1];
            }
            else if (j==0) {
                res[i][j] = im[i][j] + res[i-1][j];
            } else {
                res[i][j] = im[i][j] + res[i-1][j] + res[i][j-1] - res[i-1][j-1];
            }
        }
    }
    return res;
}

double fastsum(Mat imint, int l, int t, int w, int h) {
    int bottom = t+h+1;
    int right = l+w-1;
    int left = l;
    int top = t;

    if (left==0 && top==0) {
        return imint[bottom][right];
    }
    else if (top==0) {
        return imint[bottom][right] - imint[bottom][left-1];
    }
    else if (left==0) {
        return imint[bottom][right] - imint[top-1][right];
    } else {
        return imint[bottom][right] - imint[top-1][right] - imint[bottom][left-1] + imint[top-1][left-1];
    }
}

double fashmask(Mat imint, int left, int top, int size, int kernal) {
    if (kernal==KERNAL_DY) {
        int h = size/3;
        int w = h*2 - 1;
        int l = left + (size - w)/2;
        return fastsum(imint, l, top, w, h) - 2*fastsum(imint, l, top+h, w, h) + fastsum(imint, l, top+2*h, w, h);
    }
    else if (kernal==KERNAL_DX) {
        int w = size/3;
        int h = w*2 - 1;
        int t = top + (size - h)/2;
        return fastsum(imint, left, t, w, h) - 2*fastsum(imint, left+w, t, w, h) + fastsum(imint, left+2*w, t, w, h);
    } else {
        int w = size/3;
		int h = w;
		int l = left + (size - 2*w - 1)/2;
		int t = top + (size - 2*h - 1)/2;
		return fastsum(imint, l, top+1, w, h) - fastsum(imint, l+w+1, t, w, h) - fastsum(imint, l, t+h+1, w, h) + fastsum(imint, l+w+1, t+h+1, w, h);
    }
}

double matmin(Mat mat) {
    int h = mat.size();
    int w = mat[0].size();
    double m = mat[0][0];

    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            if (m>mat[i][j]) m = mat[i][j];
        }
    }

    return m;
}

double matmax(Mat mat) {
    int h = mat.size();
    int w = mat[0].size();
    double m = mat[0][0];

    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            if (m<mat[i][j]) m = mat[i][j];
        }
    }

    return m;
}

void normalize(Mat &mat) {
    int h = mat.size();
    int w = mat[0].size();

    double mx = matmax(mat);
    double mn = matmin(mat);

    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            mat[i][j] = (mat[i][j]-mn)/(mx-mn);
        }
    }
}

Mat convolute(Mat imint, int size=9, int kernal=KERNAL_DX) {
    int h = imint.size();
    int w = imint[0].size();
    int skip = size/2;

    Mat res = createMat(h, w, 0);

    for (int i=skip;i<h-skip;i++) {
        for (int j=skip;j<w-skip;j++) {
            res[i][j] = fastmask(imint, j-skip, i-skip, size, kernal);
        }
    }

    normalize(res);

    return res;
}

int scale2size(double scale) {
    return (int)(scale/1.2 * 9);
}

double size2scale(int size) {
    return round(size/9.0 * 1.2 * 10)/10;
}

Mat discriminant(Mat imint, int size, double w=0.9) {
    Mat dx = convolute(imint, size, KERNAL_DX);
    Mat dy = convolute(imint, size, KERNAL_DY);
    Mat dxy = convolute(imint, size, KERNAL_DXY);

    int h = dx.size();
    int w = dx[0].size();

    Mat res = createMat(h, w, 0.0);

    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            res[i][j] = dx[i][j] * dy[i][j] - pow(w*dxy[i][j], 2.0);
        }
    }

    return res;
}
