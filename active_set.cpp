#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>

Eigen::VectorXf arange(double low, double high, double step, bool with_last = false)
{
    high -= (with_last) ? 0 : step;
    int N = static_cast<int>(std::floor((high - low) / step) + 1);
    return Eigen::VectorXf::LinSpaced(N, low, high);
}

class Active_set{

    public:
        Eigen::MatrixXf H; Eigen::MatrixXf A;
        Eigen::VectorXf f; Eigen::VectorXf b;
        
        // Dual Formulation of the Problem
        Eigen::VectorXf W; Eigen::VectorXf W_hat;
        Eigen::VectorXf lambda; Eigen::MatrixXf R;

        Eigen::MatrixXf M; 
        Eigen::VectorXf d;
        Eigen::VectorXf v;

        //Primal Variables
        Eigen::VectorXf x; 

    Active_set(Eigen::MatrixXf H, Eigen::MatrixXf A, Eigen::VectorXf f, Eigen::VectorXf b){
        
        this->H = H; this->A = A;
        this->b = b; this->f = f;

        Eigen::LLT<Eigen::MatrixXf> llt(this->H);
        Eigen::MatrixXf L = llt.matrixL();
        this->R = L.transpose();

        this->M = this->A * this->R.inverse();
        Eigen::MatrixXf Rt = this->R.transpose(); 
        this->v = Rt.inverse() * this->f;
        this->d = this->b + this->M * this->v;

    }

    void solve(){
    
        this->W = Eigen::VectorXf{0};
        std::cout<<this->W<<std::endl;
        this->W_hat = arange(1, this->M.rows(),true);
        std::cout<<this->W_hat<<std::endl;
        this->lambda = Eigen::VectorXf::Zero(this->M.rows());
        this->lambda[0] = 3;

        int k = 0;

        while(true){
            
            std::cout<<this->W<<std::endl;   
            Eigen::MatrixXf M_k = this->M(Eigen::placeholders::all, this->W);
            std::cout<<M_k<<std::endl;
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

    Eigen::VectorXf f{{3.0, 2.0, 3.0}};
    Eigen::VectorXf b{{3.0, 2.0, -2.0}};

    Active_set Prob(H, A, f, b);

    Prob.solve();

    return 0;
}
