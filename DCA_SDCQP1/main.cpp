/*--------------------------------------------
# Author		: TRAN VU HOANG
# Start date	: 09-04-2025
# Start update	: 
# Language		: Cplusplus
# Version		: 1.00
# Subject		: Trust-Region DC Algorithm
# Name			: main.cpp
---------------------------------------------*/

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

using namespace std;

// --------------------- CONFIGURATION ---------------------

const double EPSILON = 1e-6;     // độ chính xác nhỏ
const int MAX_ITER = 1000;       // số vòng lặp tối đa
const int L0 = 5;                // số bước DCA mỗi vòng lặp
const double ETA1 = 0.1, ETA2 = 0.9;
const double GAMMA1 = 0.5, GAMMA2 = 0.9;
const double INITIAL_DELTA = 1.0;

// --------------------- VECTOR TOOL ---------------------

typedef vector<double> Vec;

// Phép cộng giữa hai vector
Vec operator+(const Vec& a, const Vec& b) {
    Vec res(a.size());

    for (size_t i = 0; i < a.size(); i++) 
        res[i] = a[i] + b[i];
    return res;
}

// Phép trừ giữa hai vector
Vec operator-(const Vec& a, const Vec& b) {
    Vec res(a.size());

    for (size_t i = 0; i < a.size(); i++) 
        res[i] = a[i] - b[i];
    return res;
}

// Phép nhân một scalar với một vector
Vec operator*(double scalar, const Vec& v) {
    Vec res(v.size());

    for (size_t i = 0; i < v.size(); i++) 
        res[i] = scalar * v[i];
    return res;
}

// Tính chuẩn 2 (Euclidean norm) của một vector
double norm(const Vec& v) {
    double sum = 0;

    for (double val : v) 
        sum += val * val;
    return sqrt(sum);
}

// --------------------- MAIN ALGORITHM 1 ---------------------

Vec trust_region_dca(
    const function<double(const Vec&)>& f,
    const function<Vec(const Vec&)>& grad_f,
    const function<Vec(const Vec&)>& hess_diag,  // diag of Hessian approx
    const function<Vec(const Vec&)>& projection_C,
    Vec x0
) {
    Vec xk = x0;
    double delta_k = INITIAL_DELTA;

    for (int k = 0; k < MAX_ITER; ++k) {
        Vec d_k(xk.size(), 0.0); // d^0 = 0

        double rho_k = norm(hess_diag(xk)) + EPSILON;

        // ----- DCA inner loop -----
        Vec d = d_k;

        for (int l = 0; l < L0; ++l) {
            Vec grad_model = grad_f(xk + d); // ∇m(xk + d, xk)
            Vec q = rho_k * d - grad_model;
            Vec new_d = projection_C(xk - (1.0 / rho_k) * q); // projected step
            d = new_d;
        }
        d_k = d;

        // ----- Đánh giá bước thử -----
        Vec x_trial = xk + d_k;
        double f_xk = f(xk);
        double f_trial = f(x_trial);

        // mô hình tại xk
        double model_decrease = -0.5 * rho_k * norm(d_k) * norm(d_k);
        double actual_decrease = f_xk - f_trial;
        double tau_k = actual_decrease / (-model_decrease + EPSILON); // tránh chia 0

        // ----- Chấp nhận hay từ chối bước đi -----
        if (tau_k >= ETA1) {
            xk = x_trial;
        }

        // ----- Step 3: Cập nhật delta_k -----
        if (tau_k >= ETA2) {
            delta_k *= 1.5;
        }
        else if (tau_k >= ETA1) {
            delta_k *= GAMMA2;
        }
        else {
            delta_k *= GAMMA1;
        }

        // ----- Kiểm tra hội tụ -----
        if (norm(d_k) < EPSILON) {
            std::cout << "Converged at iteration " << k << std::endl;
            break;
        }
    }

    return xk;
}