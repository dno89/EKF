/**
 * \file types.h
 * \author Daniele Molinari -- 238168
 * \version 1.0
 */

#pragma once

////include
//SLAM
#include "core.h"
//std
#include <stdexcept>


namespace EKF {
    
    ////support classes
    /**
     * @brief a class that incapsulate the vehicle model and its Jacobian
     */
    class Model {
    public:
        ////typedef
        typedef VectorType StateType;
        typedef VectorType ControlInputType;
        /**
         * @note the first argument is the vehicle state, the second is the control input
         */
        typedef VectorType (*ModelFunction)(const StateType&, const ControlInputType&);
        typedef MatrixType (*ModelJacobian)(const StateType&, const ControlInputType&);
        
        ////constructor
        /**
         * @brief proper initialization for the structure
         * @p f the vehicle model function that update the state in function of previous state and control input
         * @p df_dxv the Jacobian of @p f computed wrt the vehicle state Xv
         */
        Model(ModelFunction f, ModelJacobian df_dx) : m_F(f), m_dF_dX(df_dx) {
            if(!(m_F && m_dF_dX)) {
                throw std::runtime_error("VehicleModel::VehicleModel(VehicleFunction,VehicleJacobian) ERROR: f and df_dxv must be non-NULL!\n");
            }
        }
        /**
         * @brief empty constructor
         */
        Model() /*: m_F(nullptr), m_dF_dXv(nullptr)*/ {}
        /**
         * @brief default copy constructor
         */
        Model(const Model&) = default;
        /**
         * @brief default copy operator
         */
        Model& operator=(const Model&) = default;
        
        /**
         * @brief check whether the object has been properly initialized
         */
        operator bool() const {
            return m_F && m_dF_dX;
        }
        
        /**
         * @note Wrapper for F and dF_dXv function
         */
        VectorType F(const VectorType& Xv, const VectorType& U) const {
             if(!(m_F && m_dF_dX)) {
                throw std::runtime_error("VehicleModel::F ERROR: structure not initialized\n");
            }
            
            return (*m_F)(Xv, U);
        }
        MatrixType dF_dX(const VectorType& Xv, const VectorType& U) const {
             if(!(m_F && m_dF_dX)) {
                throw std::runtime_error("VehicleModel::F ERROR: structure not initialized\n");
            }
            
            return (*m_dF_dX)(Xv, U);
        }
        
    private:
        ////data
        ModelFunction m_F = nullptr;
        ModelJacobian m_dF_dX = nullptr;
    };
    
    /**
     * @class ProprioceptiveModel a class that incapsulate the vehicle model and its Jacobian
     */
    class ObservationModel {
    public:
        ////typedef
        typedef VectorType StateType;
        typedef VectorType ControlType;
        typedef VectorType ObservationType;
        /**
         * @note the first argument is the vehicle state, the second is the control input
         */
        typedef ObservationType (*ObservationFunction)(const StateType&, const ControlType&);
        typedef MatrixType (*ObservationJacobian)(const StateType&, const ControlType&);
        typedef VectorType (*DifferenceFunction)(const ObservationType&, const ObservationType&);
        
        ////constructor
        /**
         * @brief proper initialization for the structure
         * @p h the observation model function that gives the observation given the vehicle and landmark state
         * @p dh_dxv the Jacobian of @p h computed wrt the vehicle state Xv
         * @p dh_dxm the Jacobian of @p h computed wrt the landmark state Xm
         * @p distance a function that, given 2 perceptions returns a vector representing the distance, component by component, between the two.
         */
        ObservationModel(ObservationFunction h, ObservationJacobian dh_dxv, DifferenceFunction difference = Models::DefaultDifference) : m_H(h), m_dH_dX(dh_dxv), m_difference(difference) {
            check("ProprioceptiveModel::ProprioceptiveModel(ObservationFunction,ObservationJacobian,DifferenceFunction) ERROR: h and dh_dxv must be non-NULL!\n");
        }
        /**
         * @brief empty constructor
         */
        ObservationModel() {}
        /**
         * @brief default copy constructor
         */
        ObservationModel(const ObservationModel&) = default;
        /**
         * @brief default copy operator
         */
        ObservationModel& operator=(const ObservationModel&) = default;
        
        bool operator==(const ObservationModel& m) {
            return  m_H == m.m_H &&
                    m_dH_dX == m.m_dH_dX;
        }
        
        /**
         * @brief check whether the object has been properly initialized
         */
        operator bool() const {
            return m_H && m_dH_dX && m_difference;
        }
        
        /**
         * @brief compare operator required by the std::map
         */
        friend bool operator<(const ObservationModel& m1, const ObservationModel& m2) {
            return m1.m_H < m2.m_H;
        }
        
        /**
         * @note Wrapper for H, dH_dXv and dH_dXm functions
         */
        ObservationType H(const StateType& Xv, const ControlType& U) const {
            check("ProprioceptiveModel::H ERROR: functions not initialized\n");
            return (*m_H)(Xv, U);
        }
        MatrixType dH_dX(const StateType& Xv, const ControlType& U) const {
            check("ProprioceptiveModel::dH_dXv ERROR: functions not initialized\n");
            return (*m_dH_dX)(Xv, U);
        }
        VectorType Difference(const ObservationType& v1, const ObservationType& v2) const {
            check("ProprioceptiveModel::Difference ERROR: functions not initialized\n");
            return (*m_difference)(v1, v2);
        }
    private:
        ////data
        ObservationFunction m_H = nullptr;
        ObservationJacobian m_dH_dX = nullptr;
        DifferenceFunction m_difference = nullptr;
        
        ////function
        void check(const char* str) const {
            if(!(m_H && m_dH_dX && m_difference)) {
                throw std::runtime_error(str);
            }
        }
    };
    
    /**
     * @brief simple struct that contains a perception and the associated covariance.
     */
    struct Observation {
        Observation() {}
        Observation(const VectorType& z, const MatrixType& pz, const ObservationModel& pm) : Z(z), Pz(pz), OM(pm)
            {} 
        
        //the raw perception
        VectorType Z;
        //the covariance matrix for this perception
        MatrixType Pz;
        //the landmark model
        ObservationModel OM;
        //the accumulated size: used by the engine
        int AccumulatedSize;
    };
}