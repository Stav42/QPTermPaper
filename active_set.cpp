#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>

class Active_set{

    public:
        Eigen::MatrixXf H; Eigen::MatrixXf A;
        Eigen::VectorXf f; Eigen::VectorXf b;
        
        Eigen::VectorXf W; Eigen::VectorXf W_hat;
        Eigen::VectorXf lambda; Eigen::MatrixXf R;

        Eigen::MatrixXf M; 
        Eigen::VectorXf d;
        Eigen::VectorXf v;

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
    std::cout<<Prob.M<<std::endl;

    return 0;
}
