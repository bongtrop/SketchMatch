#include <vector>
#include <cmath>
#include <iostream>
#include <boost/python.hpp>

using namespace std;
namespace py = boost::python;

typedef std::vector< std::vector<double> > Mat;
typedef std::vector<double> Vec;

const int KERNAL_DX = 0;
const int KERNAL_DY = 1;
const int KERNAL_DXY = 2;

struct Keypoint {
    double scale;
    int x;
    int y;
    Vec des;
};

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

double fastmask(Mat imint, int left, int top, int size, int kernal) {
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

Mat discriminant(Mat imint, int size, double weight=0.9) {

    Mat dx = convolute(imint, size, KERNAL_DX);
    Mat dy = convolute(imint, size, KERNAL_DY);
    //Mat dxy = convolute(imint, size, KERNAL_DXY);

    int h = dx.size();
    int w = dx[0].size();

    Mat res = createMat(h, w, 0.0);

    /*
    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            res[i][j] = dx[i][j] * dy[i][j] - pow(weight*dxy[i][j], 2.0);
        }
    }
    */
    return res;
}

vector<Keypoint> nonmaxima(vector<Mat> octave, Vec scales, double th = 0.6) {
    int n = octave.size();

    vector<Keypoint> keypoints;

    for (int o=0;o<n;o++) {
        Mat im = octave[o];

        int h = im.size();
        int w = im[0].size();

        for (int i=0;i<h;i++) {
            for (int j=0;j<w;j++) {
                if (im[i][j]>th) {
                    int top = max(0, i-1);
                    int bottom = min(h, i+2);
                    int left = max(0, j-1);
                    int right = min(w, j+2);

                    double m = im[i][j];

                    if (o>0) {
                        Mat prev = octave[o-1];

                        for (int ii=top;ii<bottom;ii++) {
                            for (int jj=left;jj<right;jj++) {
                                if (m>prev[ii][jj]) m = prev[ii][jj];
                            }
                        }
                    }

                    if (o<n-1) {
                        Mat nxt = octave[o+1];

                        for (int ii=top;ii<bottom;ii++) {
                            for (int jj=left;jj<right;jj++) {
                                if (m>nxt[ii][jj]) m = nxt[ii][jj];
                            }
                        }
                    }

                    if (im[i][j]==m) {
                        Keypoint keypoint;
                        keypoint.x = j;
                        keypoint.y = i;
                        keypoint.scale = scales[0];
                        keypoints.push_back(keypoint);
                    }
                }
            }
        }
    }
    return keypoints;
}

vector<Keypoint> detect(Mat im, int octave=3, int period=4, double scale=1.2) {
    Mat imint = imintegral(im);
    int s = scale2size(scale);
    int add = 6;

    vector<Keypoint> keypoints;

    for (int o=0;o<octave;o++) {
        vector<Mat> ims;
        Vec scales;
        int ss = s;

        for (int i=0;i<period;i++) {
            ims.push_back(discriminant(imint, ss));
            scales.push_back(ss);
            ss+=add;
        }

        vector<Keypoint> res = nonmaxima(ims, scales);
        keypoints.insert(keypoints.end(), res.begin(), res.end());
        s+=add;
        add*=2;
    }

    return keypoints;
}

Mat list2mat(py::list pyim) {
    int h = py::extract<int>(pyim.attr("__len__")());
    int w = py::extract<int>(pyim[0].attr("__len__")());

    cout << h << endl;
    cout << w << endl;

    Mat im = createMat(h, w, 0.0);

    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++) {
            im[i][j] = py::extract<double>(pyim[i][j]);
        }
    }

    return im;
}

py::list Py_detect(py::list pyim, int octave=3, int period=4, double scale=1.2) {

    Mat im = list2mat(pyim);
    vector<Keypoint> keypoints = detect(im, octave, period, scale);

    py::list res;


    int n = keypoints.size();
    for (int i=0;i<n;i++) {
        Keypoint keypoint = keypoints[i];
        py::dict pykeypoint;
        pykeypoint["scale"] = keypoint.scale;
        pykeypoint["position"] = py::make_tuple(keypoint.x, keypoint.y);

        res.append(pykeypoint);
    }


    return res;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(fastusurf) {
    def("detect", Py_detect);
}
