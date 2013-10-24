#include <EKF/EKF.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

using namespace std;
using namespace EKF;

/**
 * Noise generation
 */
default_random_engine dre(0);
//observation
const double ob_sigma = 0.1;
normal_distribution<double> ob_noise(0.0, ob_sigma);
//state
const double state_sigma = 0.5;
normal_distribution<double> state_noise(0.0, state_sigma);

/**
 * The state is X = (x), U = (u)
 */
VectorType F(const VectorType& X, const VectorType& U) {
//     cout << "F called" << endl;
    
    VectorType res(1);
//     res << exp(-X[0]/100.0)*U[0];
    res << pow(X[0], 1.5) + U[0];
    
    return res;
}

MatrixType dF_dX(const VectorType& X, const VectorType& U) {
//     cout << "dF_dX called" << endl;
    
    MatrixType res(1, 1);
//     res << -1.0/100.0 * exp(-X[0]/100.0)*U[0];
    res << 1.5*pow(X[0], -0.5);
    
    return res;
}

const Model M1(F, dF_dX);

/**
 * The observation is Z=(x)
 */
VectorType H(const VectorType& X, const VectorType& U) {
//     cout << "H called" << endl;

    VectorType res(1);
    res << X[0];
    
    return res;
}

MatrixType dH_dX(const VectorType& X, const VectorType& U) {
//     cout << "dH_dX called" << endl;
    
    MatrixType res(1, 1);
    res << 1;
    
    return res;
}

/**
 * Noisy functions
 */
VectorType noisy_H(const VectorType& X, const VectorType& U) {
    VectorType res(1);
    res << X[0] + ob_noise(dre);
    
    return res;
}

VectorType noisy_F(const VectorType& X, const VectorType& U) {
    VectorType res(1);
//     res << cos(X[0]) - sin(X[0]) + U[0] + state_noise(dre);
    res << pow(X[0], 1.5) + U[0] + state_noise(dre);;

    
    return res;
}

const ObservationModel OM1(H, dH_dX);

double control_generator(unsigned int tick) {
    return 0.06*sin(tick/1000 * M_PI);
}

int main(int argc, char **argv) {
    std::cout << "EKF Engine test!" << std::endl;
    const unsigned int TOTAL_TICK = 10000;
    
    VectorType X(1);
    X << 1;
    
    VectorType initial_estimation(X);
    initial_estimation[0] += ob_noise(dre);
    
    MatrixType Q(1, 1);
    Q <<    state_sigma*state_sigma;
    MatrixType R(1, 1);
    R <<    ob_sigma*ob_sigma;
    
    EKFEngine e;
    e.Setup(initial_estimation, Q, M1);
    
    ofstream unfiltered("/tmp/unfiltered_state.dat");
    ofstream filtered("/tmp/filtered_state.dat");
    
    for(unsigned int ii = 0; ii < TOTAL_TICK; ++ii) {
        VectorType control(1);
        control << control_generator(ii);
        //update the state
        X = noisy_F(X, control);
        
        //the predict phase
        e.Predict(control, Q);
        e.Predict(control, Q);
        e.Predict(control, Q);
        
        Observation o(noisy_H(X, control), R, OM1);
        std::vector<Observation> obs;
        obs.push_back(o);
        
        //the update phase
        e.Update(obs, control);
        
        unfiltered << o.Z[0] << endl;
        filtered << e.GetStateEstimation()[0] << endl;
        
        cerr << '+';
    }
    
    return 0;
}
