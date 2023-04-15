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

        Eigen::MatrixXf M; 
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

    Eigen::MatrixXf replace_index(Eigen::MatrixXf lamda_k, Eigen::MatrixXf 

    void solve(){
    
        this->W.push_back(0);
        this->W.push_back(1);
        // std::cout<<this->M;
        this->W_hat = arange(2, this->M.rows());
        // std::cout<<this->W_hat<<std::endl;
        this->lambda = Eigen::VectorXf::Zero(this->M.rows());
        this->lambda[0] = 3;

        int k = 0;

        while(true){
            // std::cout<<"Working till here"<<std::endl; 
            Eigen::MatrixXf M_w = this->Splice_Index(this->W, this->M);
            Eigen::MatrixXf M_w_hat = this->Splice_Index(this->W_hat, this->M);

            Eigen::MatrixXf d_w = this->Splice_Index(this->W, this->d);
            Eigen::MatrixXf d_w_hat = this->Splice_Index(this->W_hat, this->d);
            
            Eigen::MatrixXf MMt = M_w * M_w.transpose();
            Eigen::MatrixXf lamda_k = this->lambda

            if(MMt.determinant()!=0){
                // Non-Singular
                lamda_w_k = MMt.inverse()*(-1*d_w);
                lamda_k = replace_index(lamda_k, lamda_w_k, this->W);
            }

            else{
                // Singular
                //
            }

            break;
        }

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

    return 0;
}
