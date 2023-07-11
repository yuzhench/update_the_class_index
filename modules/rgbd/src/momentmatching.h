#include <iostream>
#include<fstream>
#include <cmath>
#include <vector>
#include <array>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues> 
#include <eigen3/unsupported/Eigen/SpecialFunctions>


using namespace std;
using namespace Eigen;


class momentmatching 
{
public:
    static constexpr int num_classes = 7; 
    // param struct
    struct Params {
        Array<double, num_classes,1> a;
        Array<double, num_classes,1> alphas;
        Array<double, num_classes,1> kappas;
        Array<double, num_classes,1> betas;
        Array<double, num_classes,1> gammas;
    }post_params, prior_params, temp_params;

    Array<double , num_classes, 1> c_array;

    Params & next_params = prior_params;
    
    Matrix<double , num_classes, 2> & input_dataSetRef;
    Matrix<double , num_classes, 1> & input_aRef;

    vector<double>& measurements ;

    Matrix<double, num_classes, num_classes > exp_weigh_full;  
    Matrix<double, num_classes, num_classes >  exp_weigh_sq_full; 
    Matrix<double, num_classes, num_classes >  exp_mu_full ;
    Matrix<double, num_classes, num_classes >  exp_mu_lamda_sq_full; 
    Matrix<double, num_classes, num_classes > exp_lambda_full ;
    Matrix<double, num_classes, num_classes > exp_lambda_sq_full;

    Array<double, num_classes,1> exp_weigh;  
    Array<double, num_classes,1>  exp_weigh_sq; 
    Array<double, num_classes,1>  exp_mu ;
    Array<double, num_classes,1>  exp_mu_lamda_sq; 
    Array<double, num_classes,1> exp_lambda ;
    Array<double, num_classes,1> exp_lambda_sq;

    // sets prior from input dataset
    void initialize_prior();

    //builds posterior parameters
    void analytical_posterior(double &meaurement);
    
    //helper function to calculate moments
    void swap_elementInParams(int i);

    // evaluates moments can calulates the moment matched posterior
    void evaluate_moments();

    // updates the dataset and weights (mu, sigmas and a) directly into the input matrix
    void update_aAndDataset();

    //helper function to print params
    void print_params(Params param, string name)
    {
        cout<<param.a<< " "<< name <<" final a"<<endl;
        cout<<param.alphas<< " "<< name <<" final alphas"<<endl;
        cout<<param.kappas<< " "<< name <<" final kappas"<<endl;
        cout<<param.betas<< " "<< name <<" final betas"<<endl;
        cout<<param.gammas<< " "<< name <<" final gammas"<<endl;
    };

    // constructor which calls the necessary functions. Updates the input dataset itself, so no explicit output
    momentmatching(Matrix<double , num_classes, 2> & input_dataSet, Matrix<double , num_classes, 1> & input_a, vector<double>& meas): input_dataSetRef{input_dataSet}, input_aRef{input_a} , measurements(meas)
    {
        initialize_prior();

        for(double measure :measurements)
        {   
            analytical_posterior(measure);
            evaluate_moments();

        };
        update_aAndDataset();
        print_params(next_params, "next");
    };

};