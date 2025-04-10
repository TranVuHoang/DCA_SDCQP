#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>
#include <cassert>

typedef std::vector<double> Vec;

// ========== Toán tử vector cơ bản ==========
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

// ========== Chiếu lên tập Dₖ (ball) ==========
Vec project_onto_ball(const Vec& d, double radius) {
    double nrm = norm(d);
    if (nrm <= radius) return d;
    return (radius / nrm) * d;
}

// ========== Trust-Region DC Algorithm ==========
Vec trust_region_dc_algorithm(
    std::function<double(const Vec&)> f,
    std::function<Vec(const Vec&)> grad_f,
    std::function<double(const Vec&)> model_m,
    std::function<Vec(const Vec&)> grad_m,
    std::function<double(const Vec&)> hess_norm,
    std::function<Vec(const Vec&)> project_C,
    Vec x0,
    double delta0,
    double epsilon,
    int l0,
    double eta1,
    double eta2,
    double gamma1,
    double gamma2,
    int max_iter = 1000,
    double tol = 1e-6
) {
    Vec xk = x0;
    double delta_k = delta0;

    for (int k = 0; k < max_iter; ++k) {
        // === Step 1: DCA to solve trust-region subproblem ===
        Vec d = Vec(xk.size(), 0.0);  // d^0 := 0
        double rho = hess_norm(xk) + epsilon;

        for (int l = 0; l < l0 - 1; ++l) {
            Vec q = rho * d - grad_m(xk + d);  // q_l^k
            Vec z = (1.0 / rho) * q;
            Vec d_next = project_onto_ball(z, delta_k);  // chiếu lên D_k
            d = d_next;
        }
        Vec dk = d;

        // === Step 2: Trial point acceptance ===
        double fxk = f(xk);
        double fxk_dk = f(xk + dk);
        double model_xk = model_m(xk);
        double model_xk_dk = model_m(xk + dk);

        double tau = (fxk - fxk_dk) / std::max(model_xk - model_xk_dk, 1e-10);

        if (tau >= eta1) {
            xk = xk + dk;
        }

        // === Step 3: Update trust-region radius ===
        if (tau >= eta2) {
            delta_k *= 1.5;
        }
        else if (tau >= eta1) {
            delta_k *= gamma2;
        }
        else {
            delta_k *= gamma1;
        }

        // === Step 4: stopping condition ===
        if (norm(dk) < tol) {
            std::cout << "Converged at iteration " << k << std::endl;
            break;
        }
    }

    return xk;  // Output: x* ≈ xk
}