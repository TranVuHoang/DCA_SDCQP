/*-----------------------------------------------------
# Name file : calculations.h
# Subject   : Định nghĩa class: calculation
------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include "Eigen/Dense" // thư viện để tính giá trị riêng nhỏ nhất

using namespace Eigen;
using namespace std;

// Khai báo class: calculation dùng để định nghĩa các hàm tính toán ✅
class calculation
{
public:
    // 1. Khai báo hàm: Tích vô hướng của 2 vector x và y ✅
    double vecvec(const vector<double>& x, const vector<double>& y);

    // 2. Khai báo hàm: Chuẩn 2 của 1 vector ✅
    double norm(const vector<double>& x);

    // 3. Khai báo hàm: Nhân ma trận với vector ✅
    vector<double> matvec(const vector<vector<double>>& M, const vector<double>& x);

    // 4. Khai báo hàm: Cộng 2 vector(tính tổng 2 vector)
    vector<double> operatorPlus(const vector<double>& x, const vector<double>& y);
};

// 1. Hàm tính: Tích vô hướng của 2 vector. ✅
double calculation::vecvec(const vector<double>& x, const vector<double>& y)
{
    // kiểm tra kích thước 2 vector. ✅
    if (x.size() != y.size()) {
        cerr << "Error: vector sizes do not match in vecvec()." << endl; // in ra thông báo lỗi kích thước 2 vector không khớp nhau.
        exit(1); // dừng chương trình
    }

    double result = 0.0;
    int sizeX = x.size();

    // tính tích vô hướng của 2 vector.
    for (int i = 0; i < sizeX; i++)
        result += x[i] * y[i];

    return result; // result=x[0]⋅y[0]+x[1]⋅y[1]+⋯+x[n−1]⋅y[n−1] : kết quả tích vô hướng của 2 vector là 1 số thực.
}

// 2. Hàm tính: Chuẩn 2 của 1 vector ✅
double calculation::norm(const vector<double>& x) {
    return sqrt(vecvec(x, x));
}

// 3. Hàm tính: Nhân 1 ma trận với 1 vector ✅
vector<double> calculation::matvec(const vector<vector<double>>& M, const vector<double>& x) {
    vector<double> res(M.size(), 0.0);

    // Nhân từng hàng của ma trận với vector
    for (int i = 0; i < M.size(); i++)
        for (int j = 0; j < M[i].size(); j++)
            res[i] += M[i][j] * x[j];

    return res; // kết quả trả về 1 vector các số thực.
}

// 4. Hàm tính: Cộng 2 vector(tính tổng 2 vector) ✅
vector<double> calculation::operatorPlus(const vector<double>& x, const vector<double>& y)
{
    if (x.size() != y.size()) {
        cerr << "Error: vector size mismatch in operatorPlus." << endl; // in ra thông báo lỗi kích thước 2 vector không khớp nhau.
        exit(1);
    }

    vector<double> res(x.size(), 0.0);

    for (int i = 0; i < x.size(); i++)
        res[i] = x[i] + y[i];

    return res;
}