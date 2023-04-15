#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iterator> // for std::ostream_iterator
#include <algorithm>

std::ostream& operator<<(std::ostream& os, const std::vector<int>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
}
  

std::vector<int> arange(double low, double high)
{
    std::vector<int> W_hat;
    for(int i = low; i < high; i++){
        W_hat.push_back(i);
    }
    return W_hat;
}

class Active_set{

    public:
        Eigen::MatrixXf H; Eigen::MatrixXf A;
        Eigen::MatrixXf f; Eigen::MatrixXf b;
        
        // Dual Formulation of the Problem
        std::vector<int> W; std::vector<int> W_hat;
        Eigen::VectorXf lambda; Eigen::MatrixXf R;

        Eigen::MatrixXf M; Eigen::MatrixXf mu;
        Eigen::MatrixXf d;
        Eigen::MatrixXf v;

        //Primal Variables
        Eigen::MatrixXf x; 


        //Shapes
        int H_n, H_m;
        int A_n, A_m;
        int M_n, M_m;
        

    Active_set(Eigen::MatrixXf H, Eigen::MatrixXf A, Eigen::MatrixXf f, Eigen::MatrixXf b){
        
        this->H = H; this->A = A;
        this->b = b; this->f = f;

        Eigen::LLT<Eigen::MatrixXf> llt(this->H);
        Eigen::MatrixXf L = llt.matrixL();
        this->R = L.transpose();

        this->M = this->A * this->R.inverse();
        Eigen::MatrixXf Rt = this->R.transpose(); 
        this->v = Rt.inverse() * this->f;
        this->d = this->b + this->M * this->v;

        this->H_n = H.rows(); this->M_n = M.rows();
        this->H_m = H.cols(); this->M_m = M.cols();
        this->A_n = A.rows(); this->A_m = A.cols();

    }

    Eigen::MatrixXf Splice_Index(std::vector<int> &W, Eigen::MatrixXf &M){
        
        Eigen::MatrixXf M_w(W.size(), this->M_m);
        std::cout<<M<<std::endl;
        for(int i =0; i<W.size(); i++){
            // std::cout<<this->M.row(W[i])<<std::endl;
            M_w.row(i) = this->M.row(W[i]);
            // std::cout<<M_w.row(i)<<std::endl;
        }

        return M_w;

    }

    Eigen::MatrixXf replace_index(Eigen::MatrixXf lamda_k, Eigen::MatrixXf lamda_w_k, std::vector<int> W){
    
        int idx = 0;
        for(auto i: W){
            lamda_k.row(i) = lamda_w_k.row(idx);
            idx++;
        }

        return lamda_k;

    }

    bool all_ge_zero(Eigen::MatrixXf lamda){
    
        bool all_positive = (lamda.array() > 0).all();
        return all_positive;
    
    }

    int argmin(Eigen::MatrixXf mu, std::vector<int> W_hat){

        float min = 123123123;
        int j = -1;

        for(int i=0; i<W_hat.size(); i++){
            if(mu[W_hat[i]]<min){
                j = W_hat[i];
                min = mu[W_hat[i]];
            }
        }
        return j;
    }

    void remove(std::vector<int> &W, int k){
        auto it = std::find(W.begin(), W.end(), k);
        if (it != W.end()) { // if the value is found
            W.erase(std::remove(W.begin(), W.end(), k), W.end());
        }
    }

    std::vector<int> idx_le_zero(Eigen::MatrixXf lambda, std::vector<int> W){
    
        std::vector<int> B;
        for(int i=0; i<W.size(); i++){
            if(lambda[W[i]][0]<0)
                B.push_back(W[i]);
        }

        return B;
    }

    void fix_component(Eigen::MatrixXf lamda_k, std::vector<int> W, std::vector<int> B, Eigen::MatrixXf pk, Eigen::MatrixXf &lamda_next, std::vector<int> &W_next);

    void solve(){
    
        this->W.push_back(0);
        this->W.push_back(1);
        // std::cout<<this->M;
        this->W_hat = arange(2, this->M.rows());
        // std::cout<<this->W_hat<<std::endl;
        this->lambda = Eigen::VectorXf::Zero(this->M.rows());
        this->lambda[0] = 3;
        Eigen::MatrixXf mu_k;

        int k = 0;
        int j = -1;

        while(true){
            // std::cout<<"Working till here"<<std::endl; 
            Eigen::MatrixXf M_w = this->Splice_Index(this->W, this->M);
            Eigen::MatrixXf M_w_hat = this->Splice_Index(this->W_hat, this->M);

            Eigen::MatrixXf d_w = this->Splice_Index(this->W, this->d);
            Eigen::MatrixXf d_w_hat = this->Splice_Index(this->W_hat, this->d);
            
            Eigen::MatrixXf MMt = M_w * M_w.transpose();
            Eigen::MatrixXf lamda_k = this->lambda;

            if(MMt.determinant()!=0){
                // Non-Singular
                Eigen::MatrixXf lamda_w_k = MMt.inverse()*(-1*d_w);
                lamda_k = replace_index(lamda_k, lamda_w_k, this->W);
                
                if(all_ge_zero(lamda_k)){
                    Eigen::MatrixXf mu_k_hat = M_w_hat * M_w.transpose() * lamda_w_k + d_w_hat;
                    mu_k = replace_index(mu_k, mu_k_hat, this->W_hat);
                    this->lambda = lamda_k;

                    if(all_ge_zero(mu_k))
                        break;
                    else{
                        j = argmin(mu_k, this->W_hat);
                        this->W.push_back(j);
                        remove(this->W_hat, j);
                    }
                }
                else{
                    Eigen::MatrixXf pk = lamda_k - this->lambda;
                    std::vector<int> B = idx_le_zero(lamda_k, this->W);
                    Eigen::MatrixXf lamda_next;
                    std::vector<int> W_next;
                    fix_component(lamda_k, this->W, B, pk);
                    this->lambda = lamda_k;
                }
            }
            else{
                // Singular
                Eigen::MatrixXf mat = M_w * M_w.transpose();
                Eigen::FullPivHouseholderQR<Eigen::MatrixXf> qr(mat);
                Eigen::MatrixXf Q = qr.matrixQ();

                Eigen::MatrixXf null_space;
                if (qr.rank() < mat.cols()) {
                    null_space = Q.rightCols(mat.cols() - qr.rank());
                } else {
                    null_space = Eigen::MatrixXf::Zero(mat.rows(), 0);
                }

                Eigen::MatrixXf pk;

                for(int i=0; i<null_space.cols(); i++){
                    Eigen::MatrixXf col = null_space.col(i);
                    Eigen::MatrixXf pk_temp = replace_index(pk, col, this->W);
                    // CHECK SIZES
                    float dot = pk_temp.dot(this->d);
                    if(dot<0){
                        pk = pk_temp;  
                    }
                }

                std::vector<int> B = idx_le_zero(pk, this->W);
                fix_component(lamda_k, this->W, B, pk); 
                this->lambda = lamda_k;

            }

            k++;
        }

        Eigen::MatrixXf X = -1 * this->R.inverse() * (Splice_Index(this->W, this->M).transpose() * Splice_Index(this->W, this->lambda) + this->v);
        this->x = X;
        
    };

    void fix_component(Eigen::MatrixXf lamda_k, std::vector<int> &W, std::vector<int> B, Eigen::MatrixXf pk){
        
        int j = -1;
        float min = 123123123;
        for(int i = 0; i<B.size(); i++){
            if(-1*lamda_k[B[i]]/pk[B[i]] < min){
                min = -1*lamda_k[B[i]]/pk[B[i]];
                j = B[i];
            }
        }
        remove(W, j);
        lamda_k = lamda_k - (lamda_k[j]/pk[j])*pk;
    }
};

int main(){

    Eigen::MatrixXf H {
        {65.0, -22, 16},
        {-22,   14,  7},
        {-16,   7,   5}
    };

    Eigen::MatrixXf A {
        {1.0, 2.0, 1.0},
        {2.0, 0.0, 1.0},
        {-1.0, 2.0, -1.0}
    };

    Eigen::MatrixXf f{
                    {3.0}, 
                    {2.0}, 
                    {3.0}
    };

    Eigen::MatrixXf b{
                    {3.0}, 
                    {2.0}, 
                    {-2.0}
    };

    Active_set Prob(H, A, f, b);

    Prob.solve();
    std::cout<<Prob.x<<std::endl;

    return 0;
}
