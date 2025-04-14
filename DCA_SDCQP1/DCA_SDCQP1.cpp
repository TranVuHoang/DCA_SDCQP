#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>
#include <cassert>

#include "vec_utils.h"

using namespace std;
using Vec = vector<double>;

// ========== Trust-Region DC Algorithm ==========
Vec trust_region_dc_algorithm(
    function<double(const Vec&)> f,                  // hàm mục tiêu f(x)
    function<Vec(const Vec&)> grad_f,                // gradient ∇f(x)
    function<double(const Vec&)> model_m,            // mô hình xấp xỉ m(x; xk)
    function<Vec(const Vec&)> grad_m,                // gradient của mô hình ∇m(x; xk)
    function<double(const Vec&)> hess_norm,          // chuẩn của Hessian ∥∇²f(x)∥
    function<Vec(const Vec&)> project_C,             // phép chiếu lên tập C
    Vec x0,                                          // điểm khởi đầu x₀
    double delta0,                                   // bán kính trust-region ban đầu ∆₀
    double epsilon,                                  // ε > 0 để đảm bảo rho > 0
    int l0,                                          // số vòng lặp DCA nội bộ
    double eta1, double eta2,                        // các ngưỡng kiểm tra τₖ
    double gamma1, double gamma2,                    // hệ số điều chỉnh ∆ₖ
    int max_iter = 1000,
    double tol = 1e-6
) {
    Vec xk = x0;                  // khởi tạo điểm hiện tại
    double delta_k = delta0;      // khởi tạo bán kính vùng tin cậy
    int k = 0;

    while (true) {
        Vec d = Vec(xk.size(), 0.0);  // khởi tạo bước đi d₀ = 0
        double rho = hess_norm(xk) + epsilon;  // rhoₖ = ∥∇²f(xₖ)∥ + ε

        // === Giải bài toán con bằng DCA ===
        for (int l = 0; l < l0 - 1; ++l) {
            Vec q = rho * d - grad_m(xk + d);          // qₗ = ρ d - ∇m(xₖ + d; xₖ)
            Vec z = (1.0 / rho) * q;                   // z = q / ρ
            Vec d_next = project_onto_ball(z, delta_k); // chiếu lên ∆ₖ-ball (tập Dₖ)
            d = d_next;
        }
        Vec dk = d;  // kết quả cuối cùng của bước DCA

        // === Kiểm tra điểm thử nghiệm ===
        double fxk = f(xk);
        double fxk_dk = f(xk + dk);
        double model_xk = model_m(xk);
        double model_xk_dk = model_m(xk + dk);

        double tau = (fxk - fxk_dk) / std::max(model_xk - model_xk_dk, 1e-10);  // τₖ

        if (tau >= eta1) {
            xk = xk + dk;  // chấp nhận bước đi nếu mô hình đủ tin cậy
        }

        // === Cập nhật bán kính trust-region ===
        if (tau >= eta2) {
            delta_k *= 1.5;  // mở rộng vùng nếu bước đi rất tốt
        }
        else if (tau >= eta1) {
            delta_k *= gamma2;  // giảm nhẹ nếu tạm ổn
        }
        else {
            delta_k *= gamma1;  // thu hẹp mạnh nếu bước xấu
        }

        // === Điều kiện dừng ===
        if (norm(dk) < tol || k >= max_iter) {
            cout << "Converged at iteration " << k << endl;
            break;
        }
        k++;
    }

    return xk;  // nghiệm gần đúng x* ≈ xₖ cuối cùng
}

// ========== Problem 1: f(x) = -x₁ + 10(x₁² + x₂² − 1), C: x₁² + x₂² ≤ 1 ==========
int main() {
    Vec x0 = { 0.5, 0.5 };  // điểm khởi đầu nằm trong đĩa đơn vị

    // Hàm mục tiêu f(x)
    auto f = [](const Vec& x) {
        return -x[0] + 10 * (x[0] * x[0] + x[1] * x[1] - 1);
        };

    // Gradient của f(x)
    auto grad_f = [](const Vec& x) {
        return Vec{
            -1 + 20 * x[0],
            20 * x[1]
        };
        };

    // Chuẩn của Hessian: dùng Frobenius của ma trận hằng diag(20, 20)
    auto hess_norm = [](const Vec& x) {
        return sqrt(800.0);
        };

    // Mô hình xấp xỉ m(x; xk)
    auto model_m = [&](const Vec& y) {
        Vec d = y - x0;
        Vec g = grad_f(x0);
        return f(x0) + g[0] * d[0] + g[1] * d[1] + 0.5 * (20 * d[0] * d[0] + 20 * d[1] * d[1]);
        };

    // Gradient của mô hình m(x; xk)
    auto grad_m = [&](const Vec& y) {
        Vec d = y - x0;
        return grad_f(x0) + Vec{ 20 * d[0], 20 * d[1] };
        };

    // Phép chiếu lên tập C: đĩa đơn vị ∥x∥² ≤ 1
    auto project_C = [](const Vec& x) {
        double norm2 = x[0] * x[0] + x[1] * x[1];
        if (norm2 <= 1.0) return x;
        double scale = 1.0 / sqrt(norm2);
        return Vec{ scale * x[0], scale * x[1] };
        };

    // Gọi thuật toán với tham số cụ thể cho Problem 1
    Vec result = trust_region_dc_algorithm(
        f,
        grad_f,
        model_m,
        grad_m,
        hess_norm,
        project_C,
        x0,
        0.5,     // delta0
        1e-4,    // epsilon
        5,       // l0
        0.1,     // eta1
        0.9,     // eta2
        0.5,     // gamma1
        0.9      // gamma2
    );

    cout << "Approximate solution: (" << result[0] << ", " << result[1] << ")" << endl;
    return 0;
}