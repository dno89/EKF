/**
 * \file EKFSLAMEngine.cpp
 * \author Daniele Molinari -- 238168
 * \version 1.0
 */

////include
//SLAM
#include <EKF/EKF.h>
#include <EKF/DMDebug.h>
//std
#include <stdexcept>
#include <cassert>
#include <vector>
#include <fstream>
#include <chrono>

////DEBUG
CREATE_PUBLIC_DEBUG_LOG("/tmp/EKFEngine.log",)

using namespace EKF;
using namespace std;
using namespace Eigen;

EKFEngine::EKFEngine() {}

void EKFEngine::Setup(const VectorType& initial_state_estimation, const MatrixType& initial_covariance_estimation, const Model& model) {
        DOPEN_CONTEXT("Setup")
        
        //set the size
        m_XSize = initial_state_estimation.rows();
        
        //set the remainig parameters with some debug assertions
        assert(initial_state_estimation.rows() == m_XSize);
        m_X = initial_state_estimation;
        
        DPRINT("State initial estimation " << m_X.transpose())
        
        assert(initial_covariance_estimation.cols() == m_XSize && initial_covariance_estimation.rows() == m_XSize);
        m_Pvv = initial_covariance_estimation;
        
        DTRACE_L(m_Pvv)
        
        assert(model);
        m_Model = model;
        
        DCLOSE_CONTEXT("Setup")
        
        m_init = true;
}

void EKFEngine::Predict(const VectorType& u, const MatrixType& Q) {
    DOPEN_CONTEXT("Predict")
#ifndef NDEBUG
    auto t_start = chrono::high_resolution_clock::now();
#endif  //NDEBUG
    
    check();
    
    //use the old state to the the Taylor 1st order approximation
    MatrixType df_dXv = m_Model.dF_dX(m_X, u);
    
    DTRACE_L(u)
    DTRACE_L(Q)
    
    DTRACE(m_X.transpose())
    //predict the vehicle state
    m_X = m_Model.F(m_X, u);
    assert(m_X.rows() == m_XSize);
    //Xv is now Xv-(k)
    DPRINT("After prediction: " << m_X.transpose())
    
    //landmarks are supposed to be static: no update needed
    
    //update the covariance matrices
    //Pvv
//     DTRACE_L(df_dXv)
//     DTRACE_L(m_Pvv)
    m_Pvv = df_dXv * m_Pvv * df_dXv.transpose() + Q;
    
    assert(m_Pvv.cols() == m_XSize && m_Pvv.rows() == m_XSize);            
    DPRINT("after prediction:\n" << m_Pvv)
    
#ifndef NDEBUG
    auto dt = chrono::high_resolution_clock::now()-t_start;
    DINFO("Predict took " << chrono::duration_cast<chrono::microseconds>(dt).count() << " us.")
#endif  //NDEBUG
    
    DCLOSE_CONTEXT("Predict")
}

void EKFEngine::Update(std::vector<Observation>& observations, const VectorType& U) {
    DOPEN_CONTEXT("Update")
#ifndef NDEBUG
    auto t_start = chrono::high_resolution_clock::now();
#endif  //NDEBUG
    
    check();
    
//     const int eta_p = preprocess_proprioceptive_perceptions(proprioceptive_observations);
    const int eta_o = preprocess_observations(observations);
    const int p = observations.size();
    
    DTRACE(p)
    DTRACE(eta_o)
    DTRACE(m_X.transpose())
    
    //the innovation vector
//     VectorType std_ni(eta_p);
    VectorType ni(eta_o);
    for(int ii = 0; ii < p; ++ii) {
        DPRINT("Predicted observation : " << observations[ii].OM.H(m_X, U).transpose())
        DPRINT("Actual observation : " << observations[ii].Z.transpose())
        
        ni.segment(observations[ii].AccumulatedSize, observations[ii].Z.rows()) = observations[ii].OM.Difference(observations[ii].Z, observations[ii].OM.H(m_X, U));
    }
    
    DTRACE_L(ni)
    
    //the jacobian matrix
    MatrixType dH_dX = MatrixType::Zero(eta_o, m_XSize);
    //fill the matrix per block-row
    for(int kk = 0; kk < p; ++kk) {
        MatrixType dH_dXv = observations[kk].OM.dH_dX(m_X, U);
        
        //set it on the sparse Jacobian
        //these are the spanned rows
        for(int ii = observations[kk].AccumulatedSize; ii < observations[kk].AccumulatedSize+observations[kk].Z.rows(); ++ii) {
            //set the Xv dependent term
            //these are the spanned columns
            for(int jj = 0; jj < m_XSize; ++jj) {
                dH_dX(ii, jj) = dH_dXv(ii - observations[kk].AccumulatedSize, jj);
            }
        }
    }
    
//     DTRACE_L(m_Pvv)
//     DTRACE_L(m_Pvm)
//     DTRACE_L(m_Pmm)
//     DTRACE_L(P)
    
    MatrixType R = MatrixType::Zero(eta_o, eta_o);
    for(int ii = 0; ii < p; ++ii) {
        R.block(observations[ii].AccumulatedSize, observations[ii].AccumulatedSize, observations[ii].Z.rows(), observations[ii].Z.rows()) = observations[ii].Pz;
    }
    
    //the S matrix
    MatrixType S(eta_o, eta_o);
    S.noalias() = dH_dX * m_Pvv * dH_dX.transpose() + R;
    
//     DTRACE_L(S)
    
    //the Kalman gain
    MatrixType W(m_XSize, eta_o);
    W = m_Pvv * dH_dX.transpose() * S.inverse();
    
//     DTRACE_L(W)

    //the complete state update
    VectorType dX = W * ni;
//     DTRACE_L(dX)
    
    //update the vehicle state
    DTRACE_L(m_X)
    m_X += dX;
    DPRINT("after update:\n" << m_X)
    
    //update the total covariance matrix
    m_Pvv = m_Pvv - W * S * W.transpose();
    
//     DTRACE_L(P)
//     DTRACE_L(m_Pvv)
//     DTRACE_L(m_Pvm)
//     DTRACE_L(m_Pmm)
    
#ifndef NDEBUG
    auto dt = chrono::high_resolution_clock::now()-t_start;
    DINFO("Preassociated Update took " << chrono::duration_cast<chrono::microseconds>(dt).count() << " us.")
#endif  //NDEBUG
    
    DCLOSE_CONTEXT("Update")
}

inline void EKFEngine::check() {
    if(!m_init) throw std::runtime_error("SLAMEngine ERROR: a function has been called with the object not propertly initialize (call Setup before using the object)\n");
}

inline int EKFEngine::preprocess_observations(std::vector<Observation>& pp) const {
    int accum = 0;
    for(int ii = 0; ii < pp.size(); ++ii) {
        pp[ii].AccumulatedSize = accum;
        accum += pp[ii].Z.rows();
    }
    
    return accum;
}