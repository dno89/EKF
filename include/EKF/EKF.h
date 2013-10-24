/**
 * \file EKFSLAMEngine.h
 * \author Daniele Molinari -- 238168
 * \version 1.0
 */


#pragma once

////include
//SLAM
#include "core.h"
#include "types.h"
//std
#include <vector>

namespace EKF {
    class EKFEngine {
    public:
        ////CONSTRUCTOR
        EKFEngine();
        //no copy or assignment
        EKFEngine(const EKFEngine&) = delete;
        EKFEngine& operator=(const EKFEngine&) = delete;
        
        ////INTERFACE
        /**
         * @brief the function setup the foundamental parameters
         * @p vehicle_state_size the size of the vehicle state
         * @p initial_state_estimation the initial state
         * @p initial_covariance_estimation the initial covariance matrix of the state
         * @p vehicle_model the vehicle model according to @class VehicleModel
         */
        void Setup(const VectorType& initial_state_estimation, const MatrixType& initial_covariance_estimation, const Model& model);
        
        /**
         * @brief make the prediction step of the EKF filter
         * @p u the control input
         * @p Q the process noise covariance matrix
         */
        void Predict(const VectorType& u, const MatrixType& Q);
        
        /**
         * @brief perform update/correction based on the current perception
         * @p perceptions a vector of perception associated with known landmarks
         * @p R the TOTAL covariance matrix for all the perceptions
         */
        void Update(std::vector<Observation>& observations, const VectorType& U);
        
        /**
         * @brief get the current state estimation
         * @return the current state estimation
         */
        const VectorType& GetStateEstimation() const {
            return m_X;
        }
        
    private:
        ////DATA
        //FLAGS
        bool m_init = false;
        
        //VEHICLE STATE RELATED
        //the size of the vehicle state
        int m_XSize;
        //the actual vehicle state
        VectorType m_X;
        //the vehicle model with Jacobian
        Model m_Model;
        
        //COVARIANCE
        //the vehicle-vehicle covariance matrix
        MatrixType m_Pvv;
        
        ////SUPPORT FUNCTIONS
        void check();
        /**
         * @brief set the acculumated size for @p pp
         * @p pp the vector of associated perceptions
         * @return the total size of the observation vector
         */
        int preprocess_observations(std::vector<Observation>& pp) const;
    };
}