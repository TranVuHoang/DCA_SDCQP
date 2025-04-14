//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <functional>
//#include <algorithm>
//#include <cassert>
//
//using namespace std;
//
//using Vec = std::vector<double>;
//
//// ========= common vector =========
//Vec operator+(const Vec& a, const Vec& b) {
//    assert(a.size() == b.size());
//    Vec res(a.size());
//    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
//    return res;
//}
//
//Vec operator-(const Vec& a, const Vec& b) {
//    assert(a.size() == b.size());
//    Vec res(a.size());
//    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] - b[i];
//    return res;
//}
//
//Vec operator*(double scalar, const Vec& v) {
//    Vec res(v.size());
//    for (size_t i = 0; i < v.size(); ++i) res[i] = scalar * v[i];
//    return res;
//}
//
//double norm(const Vec& v) {
//    double sum = 0;
//    for (double val : v) sum += val * val;
//    return std::sqrt(sum);
//}
//
//Vec project_onto_ball(const Vec& d, double radius) {
//    double nrm = norm(d);
//    if (nrm <= radius) return d;
//    return (radius / nrm) * d;
//}
//
//// ========= Sequential DC Algorithm with Trust-Region and Penalty =========
//Vec sequential_dc_algorithm(
//    std::function<double(const Vec&)> phi_beta,          // phi_beta(x) = f0(x) + beta * penalty
//    std::function<double(const Vec&)> model_m_beta,      // m_beta(x, xk)
//    std::function<Vec(const Vec&)> grad_m_h,             // gradient of m_h (part of DC decomposition)
//    std::function<double(const Vec&)> penalty_function,  // p(x, xk)
//    std::function<Vec(const Vec&)> grad_m_g,             // gradient of m_g (optional, can be simplified)
//    std::function<Vec(const Vec&)> project_C,            // chiếu lên C
//    Vec x0,
//    double beta0,
//    double delta0,
//    double epsilon,
//    double delta_update,
//    int l0,
//    double eta1,
//    double eta2,
//    double gamma1,
//    double gamma2,
//    int max_iter = 1000,
//    double tol = 1e-6
//) {
//    Vec xk = x0;
//    double beta_k = beta0;
//    double delta_k = delta0;
//
//    for (int k = 0; k < max_iter; ++k) {
//        // Step 1: DCA
//        Vec d = Vec(xk.size(), 0.0);
//        for (int l = 0; l < l0; ++l) {
//            Vec q = grad_m_h(xk + d); // q_kl = ∇m_h(xk + d^l, xk)
//            Vec z = xk - (1.0 / (beta_k + epsilon)) * q;
//            Vec d_next = project_onto_ball(z - xk, delta_k); // chiếu vào Dk
//            d = d_next;
//        }
//        Vec dk = d;
//
//        // Step 2: Acceptance & penalty update
//        double tau_k = (phi_beta(xk) - phi_beta(xk + dk)) /
//            std::max(model_m_beta(xk) - model_m_beta(xk + dk), 1e-10);
//
//        if (tau_k >= eta1) {
//            xk = xk + dk;
//            double rk = std::min(penalty_function(xk + dk), penalty_function(xk));
//
//            if (beta_k < 1.0 / std::max(norm(dk), 1e-10) && rk > 0) {
//                beta_k += delta_update;
//            }
//        }
//        // else giữ nguyên xk, beta_k
//
//        // Step 3: Update trust-region radius
//        if (tau_k >= eta2) {
//            delta_k *= 1.5;
//        }
//        else if (tau_k >= eta1) {
//            delta_k *= gamma2;
//        }
//        else {
//            delta_k *= gamma1;
//        }
//
//        // Step 4: Stop nếu hội tụ
//        if (norm(dk) < tol) {
//            std::cout << "Converged at iteration " << k << std::endl;
//            break;
//        }
//    }
//
//    return xk; // nghiệm gần đúng x*
//}
//
//int main() {
//
//    return 0;
//}