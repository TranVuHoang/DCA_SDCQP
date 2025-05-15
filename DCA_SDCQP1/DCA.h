/*-----------------------------------------------------
# Name file : DCA.h
# Subject   :
------------------------------------------------------*/

#include "calculations.h"

// Khai báo class subproblem(bài toán con quadratic)
class subproblem
{
public:
    calculation cal;            // khai báo thuộc tính cal để gọi hàm tính toán. ✅
    vector<vector<double>> Q;   // Là ma trận Q trong bài toán con (quadratic): 1/2<d, <Q,d>> + <q, d>  biểu diễn thành biến u <=> 1/2 <u, <Q,u>> + <q, u> ✅
    vector<double> q;           // là vector q trong bài toán con. ✅

    // biến u trong bài toán con
    vector<double> u;
    vector<double> u_new;
    vector<double> u2;

    int n;      // kích thước của biến u trong bài toán con
    double rho; // tham số tùy chỉnh cho bài toán con
    double R;   //R là bán kính của ball được tính bằng Max Si

    // Phương thức khởi tạo subproblem
    subproblem(vector<vector<double>> Q_sub, vector<double> q_sub, int n_sub, double R_sub)
    {
        Q = Q_sub;
        q = q_sub;
        n = n_sub;
        rho = 10;
        R = R_sub;
    }

    // Hàm này tính đạo hàm của hàm H trong bài toán con của DCA để thu được u_new. ✅
    void GradH() {
        // ∇H(u) = Q·u + q
        vector<double> Qu = cal.matvec(Q, u);           // Tính: Q·u
        vector<double> gradH = cal.operatorPlus(Qu, q); // Tính: Q·u + q

        // u_new = ρ·u − ∇H(u)
        for (int i = 0; i < n; ++i)
            u_new[i] = rho * u[i] - gradH[i];           // Tính: ρ·u - ∇H
    }

    // Hàm kiểm tra điều kiện dừng cho bài toán con ✅
    bool stop(double epsilon)
    {
        vector<double> sub(n, 0.0);

        for (int i = 0; i < n; i++)
            sub[i] = u2[i] - u[i];
        double s = (cal.norm(sub) / cal.norm(u));

        return (s <= epsilon);
    }

    // Hàm để giải bài toán con bằng DCA ✅
    void solve()
    {
        // Tạo bộ sinh ngẫu nhiên ✅
        mt19937 rng(42);
        // Khai báo phân phối đều số thực trên đoạn [−1.0, 1.0]. ✅
        uniform_real_distribution<double> dist(-1.0, 1.0);

        int k_logsumexp = 10; // Number of log terms

        // khởi tạo vector u random với kích thước n. ✅
        for (auto& ui : u)
            ui = dist(rng);

        // khởi tạo vector u2 kích thước n với các giá trị 0. ✅
        u2 = vector<double>(n, 0.0);

        //for (int i = 0; i < n; i++)
        //{
        //    u2.push_back(0.0);
        //}


        // vòng lăp DCA để giải bài toán con(quadratic)
        int k = 0;

        while (true) {
            GradH();  // → tính u_new = ρu - ∇H(u) ✅

            // Chiếu u_new / rho lên ball: nếu ||u_new/rho|| > delta ⇒ scale lại           
            for (int i = 0; i < n; i++)
                u_new[i] /= rho;

            double temp = cal.norm(u_new);

            if (temp <= R)
            {
                for (int i = 0; i < n; i++)
                    u2[i] = u_new[i];
            }
            else
            {
                for (int i = 0; i < n; i++)
                    u2[i] = u_new[i] * R / temp;
            }

            if (stop(1e-5))
                break;
            u = u2;  //cập nhật u ← u2
            k++;
        }
    }
};

// Khai báo class problem (bài toán chính)
class problem
{
public:
    // các hàm tính toán
    calculation cal;

    // Problem parameters ✅
    int n, k, mc; //n: số biến x của hàm F lớn, k số hàm exp, mc là số ràng buộc

    vector<vector<double>> Q;   //D(nxn)
    vector<double> c;           //c(n)
    vector<vector<double>> a;   //a(kxn)
    vector<double> b;           //b(k)

    // ràng buộc Elip phi tuyến bậc hai (quadratic constraint): xT·Q·x + qT·x + r ≤ 0 (ct tổng quát cho mọi elipsolid)
    struct constraints {
        vector<vector<double>> Q;   // Ma trận đối xứng xác định dương
        vector<double> q;           // vector tuyến tính (dịch elip)_Trong trường hợp elipsoid đối xứng tâm tại gốc thì không dùng tới q
        double r;
    };
    vector<constraints> ellipsoids;

    /*methods*/
    void generate_data(int n_dim, int k_dim, int neclipse_dim);                     // Hàm sinh dữ liệu ngẫu nhiên cho hàm F lớn ✅

    double f(int i, const vector<double>& x_in);                                    // Hàm F
    vector<double> gradf(int i, const vector<double>& x_in);                        // Hàm tính gradientF
    vector<vector<double>> hessianf(int i, const vector<double>& x);                // Hàm tính hessainF

    double m(int i, const vector<double> x_in, const vector<double> d);
    double penalty(const vector<double> x_in);
    double mbeta(const vector<double> x_in, const vector<double> d, double beta);
    double mbetaxx(const vector<double> x_in, double beta);
    double p(const vector<double> x_in, const vector<double> d);
    double phibetak(const vector<double> x_in, const vector<double> d, double beta);
};

// Hàm tính penalty p 
double problem::p(const vector<double> x_in, const vector<double> d)
{
    vector<double> value(mc, 0);
    for (int i = 1; i <= mc; i++)
        value[i] = m(i, x_in, d);
    double maxc = value[0];
    for (int i = 1; i <= mc; i++)
        if (value[i] > maxc) maxc = value[i];
    return maxc;
}

vector<double> problem::gradf(int i, const vector<double>& x_in)
{
    if (i == 0)
    {
        vector<double> g(n, 0); //grad
        vector<double> p(k, 0.0);
        double mau = 0;
        for (int i = 0; i < k; i++)
        {
            double ax = cal.vecvec(a[i], x_in);
            p[i] = exp(ax + b[i]);
            mau += p[i];
        }
        for (int i = 0; i < k; i++) p[i] /= mau;

        vector<double> second(n, 0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                second[i] += p[j] * a[j][i];

        vector<double> Dx = cal.matvec(Q, x_in);
        for (int i = 0; i < n; i++)
            Dx[i] += c[i] - second[i];
        return Dx;  //∇f(x) = Dx + c − ∑p_i(x)*a_i       ​

    }

}

double problem::phibetak(const vector<double> x_in, const vector<double> d, double beta)
{
    return f(0, x_in) + beta * penalty(x_in);
}

double problem::penalty(const vector<double> x_in)
{
    double max_val = 0.0;
    for (auto& E : ellipsoids)
    {
        vector<double> Qd = cal.matvec(E.Q, x_in);
        double val = cal.vecvec(x_in, Qd) - E.r;
        max_val = max(max_val, val);
    }
    if (max_val < 0) max_val = 0;
    return max_val;
}

double problem::mbeta(const vector<double> x_in, const vector<double> d, double beta)
{
    return m(0, x_in, d) + beta * penalty(x_in);
}

double problem::mbetaxx(const vector<double> x_in, double beta)
{
    return f(0, x_in) + beta * penalty(x_in);
}

double problem::m(int i, const vector<double> x_in, const vector<double> d)
{
    double fd = f(i, x_in);
    vector<double> grads = gradf(i, x_in);
    vector<vector<double>> hesss = hessianf(i, x_in);  //chú ý ma trận hess tại x_k
    vector<double> Hd = cal.matvec(hesss, d);
    fd += (cal.vecvec(grads, d) + 0.5 * cal.vecvec(d, Hd));
    return fd;
}

// Hàm để sinh dữ liệu ngẫu nhiên cho hàm F lớn và ràng buộc ellip:
// F = 0.5x.Dx + c.x - log...
// contrains: ...
void problem::generate_data(int n_dim, int k_dim, int neclipse_dim)
{
    n = n_dim; k = k_dim; mc = neclipse_dim;

    mt19937 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    // khởi tạo ma trận chéo D xác định dương kích thước D(nxn). ✅
    Q = vector<vector<double>>(n, vector<double>(n, 0.0)); // Q or D?
    for (int i = 0; i < n; i++)
        Q[i][i] = 0.1 + abs(dist(rng)); // Q or D?

    // khởi tạo vector c random kích thước c(n) - Sinh ngẫu nhiên từ phân phối đều trên [−1,1]. ✅
    c = vector<double>(n);
    for (auto& ci : c)
        ci = dist(rng);

    // khởi tạo ma trận a(k x n) - Sinh ngẫu nhiên từ phân phối đều trên [−1,1]. ✅
    a = vector<vector<double>>(k, vector<double>(n));
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = dist(rng);

    // khởi tạo vector b kích thước b(n) - Sinh ngẫu nhiên từ phân phối đều trên [−1,1]. ✅
    b = vector<double>(k);
    for (auto& bi : b)
        bi = dist(rng);

    // khởi tạo ràng buộc ellip: xT.Qx + qT.x + r ≤ 0 ✅
    ellipsoids.clear();
    for (int e = 0; e < mc; e++)
    {
        constraints E;
        // khởi tạo ma trận chéo Q
        E.Q = vector<vector<double>>(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; i++)
            E.Q[i][i] = 0.01 + abs(dist(rng));

        // khởi tạo vector q
        E.q = vector<double>(n);
        for (auto& qi : E.q)
            qi = dist(rng);

        // khởi tạo hằng số s
        E.r = dist(rng) * 5;
        ellipsoids.push_back(E);
    }
}

// Hàm F0, và Fi
double problem::f(int i, const vector<double>& x_in)
{
    if (i == 0)
    {
        vector<double> Qx = cal.matvec(Q, x_in);
        double xTQx = cal.vecvec(x_in, Qx);
        double cx = cal.vecvec(c, x_in);

        vector<double> exps(k);
        for (int i = 0; i < k; i++) {
            exps[i] = cal.vecvec(a[i], x_in) + b[i];
        }

        double sum_exp = 0.0;
        for (int i = 0; i < k; i++)
            exps[i] = exp(exps[i]),
            sum_exp += exps[i];
        return 0.5 * xTQx + cx - log(sum_exp);
    }
    else
    {
        vector<double> Qx = cal.matvec(ellipsoids[i].Q, x_in);
        double xTQx = cal.vecvec(x_in, Qx);
        //double qTx = cal.vecvec(E.q, x_in); 
        return 0.5 * xTQx - ellipsoids[i].r;
    }
}

// Tính đạo hàm cấp 2 Hessian của F ✅
vector<vector<double>> problem::hessianf(int i, const vector<double>& x_in)
{
    vector<vector<double>> H(n, vector<double>(n, 0.0));  // Khởi tạo ma trận Hessian ban đầu
    // 1. Tính: (0.5 * x^T.D.x + c^T.x)" = (D * x)' = D ✅

    if (i == 0)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                H[i][j] = Q[i][j];  // D (hàm chuẩn)
        // 2. Tính log(Σ exp(aᵢ^T x + bᵢ))"
        // 2.1 Tính: aᵢ^T x + bᵢ cho mỗi i, và lưu vào mảng
        vector<double> exps(k);
        for (int i = 0; i < k; ++i) {
            exps[i] = cal.vecvec(a[i], x_in) + b[i]; // Tính aᵢ^T x + bᵢ và lưu vào exps[i]
        }

        // 2.2 Tính: Σ exp(aᵢ^T x + bᵢ)
        double sum_exp = 0.0;
        for (int i = 0; i < k; i++) {
            exps[i] = exp(exps[i]);  // Tính exp(aᵢ^T x + bᵢ) cho mỗi i
            sum_exp += exps[i]; // Σ exp(aᵢ^T x + bᵢ)
        }

        // 2.3 trừ đi tổng: ∑ s_i * A_i * A_i^T
        for (int i = 0; i < k; i++) {
            double si = exps[i] / sum_exp; // si = exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    H[r][c] -= si * a[i][r] * a[i][c];  // -∑ s_i * A_i * A_i^T
        }

        // 2.4 Tính tổng: ∑ s_i * Ai
        vector<double> weighted_sum(n, 0.0);
        for (int i = 0; i < k; i++) {
            double si = exps[i] / sum_exp; // si = exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)
            for (int j = 0; j < n; ++j)
                weighted_sum[j] += si * a[i][j]; // ∑ [exp(aᵢ^T x + bᵢ) / Σexp(aᵢ^T x + bᵢ)] * Ai
        }
        // 2.5 Cộng tích ngoài của weighted_sum: + (∑ s_i a_i)(∑ s_i a_i)^T
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                H[r][c] += weighted_sum[r] * weighted_sum[c];  // + (∑ s_i a_i)(∑ s_i a_i)^T
    }
    else
    {
        for (int t = 0; t < n; t++)
            for (int j = 0; j < n; j++)
                H[t][j] = ellipsoids[i].Q[t][j];
    }
    return H; // ma trận Hessian 
}

//=================================================================
class DCA : public problem // class DCA kế thừa class problem
{
    calculation cal; // !!!

    const double BETA0 = 10.0;
    double BETAK;
    const double DELTA = 0.8; //DELTA LÀ LƯỢNG THÊM VÀO BETAK mỗi khi update BETAK
    const double EPS = 1e-6;
    double DELTA0 = 10.0; // bán kính ball 

    const int DCA_STEPS = 5; // l° số bước lặp của step 1
    const int MAX_ITER = 1000;

    const double ETA1 = 0.001, ETA2 = 0.25;
    const double GAMMA1 = 0.5, GAMMA2 = 0.5;

    const double RHO = 10.0; // tham số rho cho bài toán QP con
    const double RHO1 = 10.0;
    const double RHO2 = 10.0;

    int l0 = 10;

    double t = 0;
    double tauk; // tôk
    int n;

    vector<vector<double>> D;

    //Biến của bài toán lớn
    vector<double> x;

    //Biến của bài toán quadratic step 1
    vector<double> d;
    vector<double> d_new;
    vector<double> d2;

    vector<double> grad;
    vector<vector<double>> hess;

    //================================
    void init();                //Hàm khởi tạo của bài toán toán chính
    void Algorithm();           // Thuật toán giải bài toán lớn 
    void QP_solve();            // Step 1
    void updatexbeta();         // Step 2
    void updateTrustRegion();   // Step 3

    void QP_init();             // Hàm khởi tạo d cho bài toán step1
    void QP_GradientH();        // Tại mỗi bước lặp ta dùng 1 DCA
    void QP_GradientG();        // Tại mỗi bước lặp ta dùng 1 DCA
    bool QP_Stop();             // Hàm này không dùng đến vì DCA chỉ chạy l0 vòng
    double QP_getObj();         // In hàm mục tiêu

    bool Stop();          //Điều kiện dừng của bài toán to


};

// Khởi tạo cho bài toán chính !!!
void DCA::init() // Khởi tạo cho x thuộc C (toàn không gian)
{
    mt19937 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    while (true)
    {
        // 1. Tạo ngẫu nhiên 1 điểm x
        for (auto& xi : x)
            xi = dist(rng);

        // 2. Kiểm tra điều kiện của 3 ellipsoid
        bool in_all = true;

        for (int i = 0; i < mc; ++i) {
            // Tính x^T Q_i x
            vector<double> Qx = cal.matvec(ellipsoids[i].Q, x); // Q[i] là ma trận Q của ellipsoid thứ i
            double xtQx = cal.vecvec(x, Qx);
            if (xtQx > ellipsoids[i].r) { // alpha[i] = r_i + s_i
                in_all = false;
                break;
            }
        }
        if (in_all) break; // x thỏa cả 3 ellipsoid
    }

    BETAK = 10;
}

// Thuật toán chính
void DCA::Algorithm()
{
    init(); // khởi tạo x°
    int step = 0;

    while (true)
    {
        // Step 1
        QP_solve(); // Giải bài toán QP con để thu được dk

        // Step 2 
        updatexbeta();

        // Step 3
        updateTrustRegion();

        // step 4
        step++;
        if (step > l0)
            break;
    }
}

// Step 1: Giải bài toán con dạng QP bằng DCA để thu đc vector dk
void DCA::QP_solve()
{
    // 1.1 Khởi tạo cho bai toán QP con
    QP_init(); // khởi tạo d°

    int k = 0;

    while (true)
    {
        if (k >= l0) break;
        QP_GradientH(); // tính d_new ✅
        QP_GradientG(); // tính d2       
        d = d2;
        k++;
    }
}

// Step 1.1 Hàm khởi tạo cho bài toán QP của step 1
void DCA::QP_init()
{
    // khởi tạo vector d kích thước n với các giá trị 0. ✅
    d = vector<double>(n, 0.0);
    //sau khởi tạo d = 0 hết
}

// 1.2 Hàm tính đạo hàm của H của bài toán chính để thu được vector d_new ✅
void DCA::QP_GradientH()
{
    // Đảm bảo d_new có đúng số phần tử 
    d_new.resize(n);

    // Tính d_new(q_k^l) = ∇mh(xK + dl_k) = [0.5 * (rho0 + beta_k * rho_xk) * ||d||²]'  = (RHO1 + beta_k * RHO2) * d
    for (int i = 0; i < n; i++)
        d_new[i] = (RHO1 + BETAK * RHO2) * d[i]; // trả về vector d_new chính là qlk ✅
}

// 1.3 Hàm tính đạo hàm của G của bài toán chính để thu được vector d_2 !!!
void DCA::QP_GradientG()
{
    //Chuẩn bị Q, q của bài quadratic: 1/2<d, <Q,d>> + <q, d>
    vector<vector<double>> Q = hess; // Q = Hess(xk)+(Ro1+βkRo2)⋅I
    vector<double> q;

    for (int i = 0; i < n; i++)
    {
        Q[i][i] += (RHO1 + BETAK * RHO2);
        q[i] = grad[i] - d_new[i];
    }
    //Khởi tạo bài toán con
    subproblem p(Q, q, n, RHO);
    //Giải bài toán con để thu được d2
    p.solve();
    d2 = p.u;
}

double DCA::QP_getObj()  // obj của bài quadratic tại điểm d
{
    double fd = 0.0;   //không cần đặt bằng f(x_k)
    vector<double> HessD = cal.matvec(hess, d);  //chú ý ma trận hess tại x_k

    fd += (cal.vecvec(grad, d) + 0.5 * cal.vecvec(d, HessD));  //chú ý véc tơ grad tại x_k

    double max_val = 0.0;
    for (auto& E : ellipsoids) {
        vector<double> Qd = cal.matvec(E.Q, d);
        double val = cal.vecvec(d, Qd) + E.r;
        max_val = max(max_val, val);
    }
    if (max_val < 0)max_val = 0;

    return fd + BETAK * max_val;
}


// Điều kiện dừng cho bài toan QP của step 1
bool DCA::QP_Stop()
{
    vector<double> sub(n, 0.0);

    for (int i = 0; i < n; i++)
        sub[i] = d2[i] - d[i];
    double s = (cal.norm(sub) / cal.norm(d));
    return (s <= EPS);
}

// Step 2 
void DCA::updatexbeta()
{
    vector<double> xplusd = cal.operatorPlus(x, d);

    tauk = (phibetak(x, d, BETAK) - phibetak(xplusd, d, BETAK)) / (mbetaxx(x, BETAK) - mbeta(x, d, BETAK));

    if (tauk >= ETA1)
    {
        double rk = min(p(x, d_new), p(xplusd, d));

        if (BETAK < 1 / cal.norm(d) && rk > 0)
            BETAK += DELTA;
        x = xplusd;
    }
}

// Step 3
void DCA::updateTrustRegion()
{
    if (tauk >= ETA2)
        DELTA_BETA += 0.1;
    else if (tauk >= ETA1 && tauk < ETA2)
        DELTA_BETA = GAMMA2 * DELTA_BETA + 0.1;
    else
        DELTA_BETA = (GAMMA1 + GAMMA2) / 2.0 * DELTA_BETA;
}
