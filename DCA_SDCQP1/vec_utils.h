#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <cassert>

using namespace std;

using Vec = vector<double>;

// ========= Toán tử vector =========
Vec operator+(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
    return res;
}

Vec operator-(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] - b[i];
    return res;
}

Vec operator*(double scalar, const Vec& v) {
    Vec res(v.size());
    for (size_t i = 0; i < v.size(); ++i) res[i] = scalar * v[i];
    return res;
}

double norm(const Vec& v) {
    double sum = 0;
    for (double val : v) sum += val * val;
    return std::sqrt(sum);
}

Vec project_onto_ball(const Vec& d, double radius) {
    double nrm = norm(d);
    if (nrm <= radius) return d;
    return (radius / nrm) * d;
}