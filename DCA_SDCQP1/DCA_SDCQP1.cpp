﻿#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include "Eigen/Dense" // thư viện để tính giá trị riêng nhỏ nhất

using namespace Eigen;
using namespace std;

// Hyperparameters ✅
const double BETA0 = 10.0;
const double DELTA0 = 1.0;

const double EPS = 1e-6;
const double DELTA_BETA = 10.0;

const int DCA_STEPS = 5; // l°
const int MAX_ITER = 1000;

const double ETA1 = 0.001, ETA2 = 0.25;
const double GAMMA1 = 0.5, GAMMA2 = 0.5;
const double RHO = 10.0;
const double RHO_0 = 10.0;

// Problem parameters ✅
int n = 4; // Dimension
int k_logsumexp = 10; // Number of log terms

// Random generator ✅
mt19937 rng(42);
uniform_real_distribution<double> dist(-1.0, 1.0);

// Data containers ✅
vector<vector<double>> D;
vector<double> c;
vector<vector<double>> A;
vector<double> b;

struct Ellipsoid {
    vector<vector<double>> Q;
    vector<double> r;
    double s;
};
vector<Ellipsoid> ellipsoids;

// Tích vô hướng của 2 vector ✅
double dot(const vector<double>& a, const vector<double>& b) {
    double res = 0.0;

    for (int i = 0; i < a.size(); ++i)
        res += a[i] * b[i];

    return res;
}

// Nhân ma trận với vector ✅
vector<double> matvec(const vector<vector<double>>& M, const vector<double>& x) {
    vector<double> res(M.size(), 0.0);
    for (int i = 0; i < M.size(); ++i)
        for (int j = 0; j < M[i].size(); ++j)
            res[i] += M[i][j] * x[j];

    return res;
}

// Chuẩn 2 của 1 vector ✅
double norm(const vector<double>& x) {
    return sqrt(dot(x, x));
}

// Tính F(x) ✅
double F(const vector<double>& x) {
    // 1. Tính phần bậc 2: 0.5 * x^T D x ✅
    vector<double> Dx = matvec(D, x);
    double quad = 0.5 * dot(x, Dx);

    // 2. Phần tuyến tính: c^T x ✅
    double linear = dot(c, x);

    // 3. Phần log-sum-exp: -log(Σ exp(aᵢ^T x + bᵢ)) ✅
    // 3.1 Tính: (aᵢ^T x + bᵢ)
    vector<double> exps(k_logsumexp);
    for (int i = 0; i < k_logsumexp; ++i) {
        exps[i] = dot(A[i], x) + b[i];
    }

    // 3.2 Tính: Σexp(aᵢ^T x + bᵢ)
    double sum_exp = 0.0;
    for (int i = 0; i < k_logsumexp; ++i)
        sum_exp += exp(exps[i]);

    // 3.3 Tính: log(Σ exp(aᵢ^T x + bᵢ))
    double logsumexp = log(sum_exp);

    return quad + linear - logsumexp; // trả về F
}

// Gradient of F(x) ✅
vector<double> gradF(const vector<double>& x) {
    // 1. Tính: (0.5 * x^T.D.x + c^T.x)' = Dx + c ✅
    vector<double> gradientF = matvec(D, x);  // D * x
    for (int i = 0; i < n; i++)
        gradientF[i] += c[i];  // + c

    // 2. Tính: (log-sum-exp)'

    // Tính: (aᵢ^T x + bᵢ) ✅
    vector<double> exps(k_logsumexp);
    for (int i = 0; i < k_logsumexp; ++i) {
        exps[i] = dot(A[i], x) + b[i];
    }

    // Tính: Σexp(aᵢ^T x + bᵢ) 
    double sum_exp = 0.0;
    for (int i = 0; i < k_logsumexp; i++)
        exps[i] = exp(exps[i]),
        sum_exp += exps[i];

    // Tính F'
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k_logsumexp; ++j) {
            gradientF[i] -= (exps[j] / sum_exp) * A[j][i];
        }
    }
    return gradientF; // trả về 1 vector số thực kích thước n
}

// Tính đạo hàm cấp 2 Hessian của F ✅
vector<vector<double>> hessianF(const vector<double>& x) {
    vector<vector<double>> H(n, vector<double>(n, 0.0));  // Khởi tạo ma trận Hessian ban đầu

    // 1. Tính: (0.5 * x^T.D.x + c^T.x)" = (D * x)' = D ✅
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            H[i][j] = D[i][j];

    // 2. Tính log(Σ exp(aᵢ^T x + bᵢ))"
    // 2.1 Tính: (aᵢ^T x + bᵢ) ✅
    vector<double> exps(k_logsumexp);
    for (int i = 0; i < k_logsumexp; ++i) {
        exps[i] = dot(A[i], x) + b[i];
    }

    // 2.2 Tính: Σexp(aᵢ^T x + bᵢ) ✅
    double sum_exp = 0.0;
    for (int i = 0; i < k_logsumexp; i++)
        exps[i] = exp(exps[i]), // exps[i] = exp(aᵢ^T x + bᵢ)
        sum_exp += exps[i]; // Σexp(aᵢ^T x + bᵢ)

    // 2.3 trừ đi tổng: ∑ s_i * A_i * A_i^T ✅
    for (int i = 0; i < k_logsumexp; i++) {
        double si = exps[i] / sum_exp; // si = exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++)
                H[r][c] -= si * A[i][r] * A[i][c]; // -∑ s_i * A_i * A_i^T
    }

    // 2.4 Tính tổng:  ∑ s_i * Ai =  ∑ [exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)] * Ai ✅
    vector<double> weighted_sum(n, 0.0);
    for (int i = 0; i < k_logsumexp; i++) {
        double si = exps[i] / sum_exp; // si = exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)
        for (int j = 0; j < n; ++j)
            weighted_sum[j] += si * A[i][j]; // ∑ [exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)] * Ai
    }

    // 2.5 Cộng tích ngoài của weighted_sum: + (∑ s_i a_i)(∑ s_i a_i)^T ✅
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            H[r][c] += weighted_sum[r] * weighted_sum[c];

    return H; // ma trận hessan 
}

// Hàm in ma trận H ✅
void print_matrix(const vector<vector<double>>& H, int rows = 10, int cols = 10) {
    cout << "Partial Hessian matrix H:" << endl;
    cout << fixed << setprecision(6); 
    for (int i = 0; i < min(rows, (int)H.size()); ++i) {
        for (int j = 0; j < min(cols, (int)H[i].size()); ++j) {
            cout << H[i][j] << "\t";
        }
        cout << endl;
    }
}

/* Tính rho_xk theo công thức (24)  ✅*/
// Hàm chuyển từ vector<vector<double>> sang Eigen::MatrixXd ✅
MatrixXd toEigenMatrix(const vector<vector<double>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    MatrixXd eigenMat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            eigenMat(i, j) = mat[i][j];
    return eigenMat;
}

// hàm tính ρ_0 ​​= max(0,−λmin​) ✅
double compute_rho0(const vector<double>& x) {
    vector<vector<double>> H_vec = hessianF(x);
    MatrixXd H = toEigenMatrix(H_vec);

    SelfAdjointEigenSolver<MatrixXd> solver(H);
    double lambda_min = solver.eigenvalues()(0);  // nhỏ nhất nằm đầu

    double epsilon = 1e-6;
    return max(0.0, -lambda_min);
}

// Xây dựng mô hình m0
double m0(const vector<double>& xk, const vector<double>& dk, const vector<double>& gradF, const vector<vector<double>>& H) {
    vector<double> Hd = matvec(H, dk);
    double gradientF = dot(gradF, dk);
    double hessainF = 0.5 * dot(dk, Hd);
    return F(xk) + gradientF + hessainF;
}

// Penalty function p ✅
double pel(const vector<double>& x) {
    double max_val = 0.0;
    for (auto& E : ellipsoids) {
        vector<double> Qx = matvec(E.Q, x);
        double val = dot(x, Qx) + dot(E.r, x) + E.s;
        max_val = max(max_val, val);
    }
    return max_val;
}

// Penalty function p+ ✅
double penalty(const vector<double>& x) {
    double max_val = 0.0;
    for (auto& E : ellipsoids) {
        vector<double> Qx = matvec(E.Q, x);
        double val = dot(x, Qx) + dot(E.r, x) + E.s;
        max_val = max(max_val, val);
    }
    return max(0.0, max_val);
}

// xây dựng mô hình mBeta
double m_beta(const vector<double>& xk, const vector<double>& d,
    const vector<double>& gradFx, const vector<vector<double>>& H,
    double f0_xk, double beta_k) {

    // Tính f_0(x_k) + ∇f_0(x_k)^T d + 0.5 * d^T H d
    vector<double> Hd = matvec(H, d);
    double linear = dot(gradFx, d);
    double quad = 0.5 * dot(d, Hd);
    double f_approx = f0_xk + linear + quad;

    // Tính p⁺(x_k + d)
    vector<double> xk_plus_d = xk;
    for (int i = 0; i < xk.size(); ++i)
        xk_plus_d[i] += d[i];
    double penalty_val = penalty(xk_plus_d);

    // Tổng lại
    return f_approx + beta_k * penalty_val;
}
// F(x) + dot(grad, d) + 0.5 * RHO * dot(d, d) + beta * penalty(x);

// phân ra DC m_g - m_h
// Xây dựng m_h
double m_h(const vector<double>& d, const vector<double>& gradFx, double rho0, double beta_k) {
    double rho_xk = 1e-6;
    double coeff = (rho0 + beta_k * rho_xk) / 2.0;
    return (rho0 + beta_k * rho_xk) / 2.0 * dot(d, d);
}

// Xây dựng m_g
double m_g(const vector<double>& xk, const vector<double>& d,
    const vector<double>& gradFx, const vector<vector<double>>& H,
    double f0_xk, double beta_k, double rho0) {

    double mh = m_h(d, gradFx, rho0, beta_k);
    double mbeta = m_beta(xk, d, gradFx, H, f0_xk, beta_k);
    return mh + mbeta;
}

// tính đạo hàm của m_h
vector<double> grad_mh(const vector<double>& d, double rho0, double beta_k, double rho_xk) {
    double coeff = rho0 + beta_k * rho_xk;
    vector<double> grad(d.size());
    for (int i = 0; i < d.size(); ++i)
        grad[i] = coeff * d[i];
    return grad;
}

// tính phiBeta ✅
double phi_beta(const vector<double>& x, double beta) {
    return F(x) + beta * penalty(x);
}

// Project onto trust region ball ✅
void project_trust_region(vector<double>& d, double delta) {
    double norm_d = norm(d);
    if (norm_d > delta) {
        for (auto& di : d)
            di *= delta / norm_d;
    }
}

// Generate random data ✅
void generate_data() {
    // khởi tạo ma trận chéo D xác định dương kích thước D(nxn)
    D = vector<vector<double>>(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        D[i][i] = 0.1 + abs(dist(rng));

    // khởi tạo vector c kích thước c(n)
    c = vector<double>(n);
    for (auto& ci : c) ci = dist(rng);

    // khởi tạo ma trận A
    A = vector<vector<double>>(k_logsumexp, vector<double>(n));
    for (int i = 0; i < k_logsumexp; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = dist(rng);

    // khởi tạo vector b kích thước b(n)
    b = vector<double>(k_logsumexp);
    for (auto& bi : b) bi = dist(rng);

    ellipsoids.clear();
    for (int e = 0; e < 3; ++e) {
        Ellipsoid E;

        // khởi tạo ma trận chéo Q
        E.Q = vector<vector<double>>(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            E.Q[i][i] = 0.01 + abs(dist(rng));

        // khởi tạo vector r
        E.r = vector<double>(n);
        for (auto& ri : E.r) ri = dist(rng);

        // khởi tạo hằng số s
        E.s = dist(rng) * 5.0;
        ellipsoids.push_back(E);
    }
}

// Hàm solveSubProblemQuadratic
vector<double> solveSubProblemQuadratic(vector<vector<double>>& Q, vector<double>& q, double rho, int n, double delta) {
    // khởi tạo vector u random với kích thước n
    vector<double> u(n);
    for (auto& ui : u) {
        ui = dist(rng);
    }

    int k = 0;
    while (true) {
        // Tính gradientH = (rho.I - Q).u - q
        vector<double> GradH(n, 0);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum -= Q[i][j] * u[j];
            sum += rho * u[i];
            GradH[i] = sum - q[i];
        }

        // Tính u_new = rho * u - GradH
        vector<double> u_new(n);
        for (int i = 0; i < n; i++)
            u_new[i] = rho * u[i] - GradH[i];

        // Chiếu u_new / rho lên ball: nếu ||u_new/rho|| > delta ⇒ scale lại
        double norm_u = norm(u_new) / rho;
        if (norm_u <= delta) {
            for (int i = 0; i < n; i++)
                u[i] = u_new[i] / rho;
        }
        else {
            for (int i = 0; i < n; i++)
                u[i] = (u_new[i] / norm_u) * delta;
        }
        if (norm(GradH) < 1e-5)
            break;

        k++;
    }
    return u; // chính là d_k
}

int main() {
    generate_data();

    // khởi tạo ngẫu nhiên x° random (-10, 10) 
    vector<double> x(n);
    for (auto& xi : x) {
        xi = dist(rng) * 10;
    }

    // in giá trị điểm khởi tạo x ban đầu
    cout << "Initial x_0: " << endl;
    for (int i = 0; i < n; i++)
        cout << x[i] << " ";
    cout << endl;

    vector<vector<double>> H = hessianF(x); // Tính Hessian tại x
    cout << "------------------------------------------------------------" << endl;
    print_matrix(H); // In ra 10×10 đầu tiên mặc định

    double Fx0 = F(x); // Lưu F(x0)
    cout << "------------------------------------------------------------" << endl;
    cout << "F(x_0) = " << Fx0 << endl;
    cout << "------------------------------------------------------------" << endl;

    double beta = BETA0;
    double delta = DELTA0;

    int total_iterations = 0;
    bool stopped_by_eps = false;

    // Tính rho_0 và rho_{x_k}
    double rho0 = compute_rho0(x);
    cout << "rho0 = " << rho0 << endl;
    double rho_xk = 1e-6;
    cout << "rho_xk = " << rho_xk << endl;

    vector<double> dk(n);
    int k = 0;
    while (true) {
        vector<double> d(n, 0.0);  // d0 ban đầu

        // step 1 tính dk bằng cách giải bài toán con dạng quadratic
        for (int l = 0; l < DCA_STEPS; l++) {
            // 1. Tính d_new(q_k^l) = ∇mh(xK + dl_k) = [0.5 * (rho0 + beta_k * rho_xk) * ||d||²]'  = (rho0 + beta_k * rho_xk) * d
            vector<double> d_new(n);
            for (int i = 0; i < n; i++)
                d_new[i] = (rho0 + beta * rho_xk) * d[i]; // trả về vector d_new chính là qlk ✅

            // 2. update lại dl bằng cách giải bài toán Min mg(xk+dl, xk) - <q, dl>
            // Tính Q =  Hess(xk)+(Ro1+βkRo2)⋅I
            vector<vector<double>> Q = hessianF(x);
            for (int i = 0; i < n; i++)
                Q[i][i] += (rho0 + beta * rho_xk);

            // Tính q = GradF(xk) - d_new
            vector<double> q = gradF(x);
            for (int i = 0; i < n; i++)
                q[i] -= d_new[i];

            //rho = 10
            int rho = 10;

            // tính dk 
            vector<double> dk = solveSubProblemQuadratic(Q, q, rho, n, delta);
            // in giá trị dk 
            cout << "------------------------------------------------------------" << endl;
            cout << "In vector dk tai vong lap thu: " << l + 1 << endl;
            for (int i = 0; i < n; i++)
                cout << dk[i] << " ";
            cout << endl << "norm_dk = " << norm(dk);
            cout << endl;
        }

        // Step 2 chấp nhận điểm thử xK và update penalty
        vector<double> x_new(n);
        for (int i = 0; i < n; i++)
            x_new[i] = x[i] + dk[i]; // xk + dk

        double phi_betak_xk = phi_beta(x, beta); // ✅
        double phi_betak_xk_dk = phi_beta(x_new, beta); // ✅

        //vector<double> grad = gradF(x);

        double m_betak_xk = F(x) + beta * penalty(x); // ✅
        double m_betak_xk_dk = F(x) + dot(gradF(x), dk) + 0.5 * dot(dk, matvec(hessianF(x), dk)) + beta * penalty(x_new);

        double tau_k = (phi_betak_xk - phi_betak_xk_dk) / max(1e-12, (m_betak_xk - m_betak_xk_dk));


        cout << "Iter " << k << ": phi = " << phi_betak_xk << ", tau_k = " << tau_k << ", beta = " << beta << ", dk = " << norm(d) << endl;

        total_iterations = k + 1;

        if (tau_k >= ETA1) {
            x = x_new;
            double r_k = min(penalty(x_new), penalty(x));

            if (beta >= 1.0 / (norm(dk) + 1e-8) || r_k <= 0)
                beta = beta;
            else
                beta += DELTA_BETA;

            if (tau_k >= ETA2) delta *= 1.5;
            else delta *= GAMMA2;
        }
        else {
            delta *= GAMMA1;
        }

        if (norm(d) < EPS) {
            stopped_by_eps = true;
            break;
        }
        k++;
    }

    // in giá trị dk 
    cout << "------------------------------------------------------------" << endl;
    cout << "In dk: " << endl;
    for (int i = 0; i < n; i++)
        cout << dk[i] << " ";
    cout << endl;
    cout << "------------------------------------------------------------" << endl;

    cout << "Optimal x found. F(x*) = " << F(x) << endl;
    cout << "Checking constraints at x*:" << endl;
    for (int j = 0; j < ellipsoids.size(); ++j) {
        vector<double> Qx = matvec(ellipsoids[j].Q, x);
        double gj = dot(x, Qx) + dot(ellipsoids[j].r, x) + ellipsoids[j].s;
        cout << "g_" << j + 1 << "(x*) = " << gj << endl;
    }

    cout << "-----------------------------------------" << endl;
    cout << "Summary for this run:" << endl;
    cout << "n = " << n << ", k = " << k_logsumexp << endl;
    cout << "F(x0) = " << Fx0 << endl;

    // in giá trị  x*
    cout << "Initial x*: " << endl;
    for (int i = 0; i < n; ++i) cout << x[i] << " ";
    cout << endl;

    cout << "F(x*) = " << F(x) << endl;
    cout << "Total iterations = " << total_iterations << endl;

    return 0;
}
