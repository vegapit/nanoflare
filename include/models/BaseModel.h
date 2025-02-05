#pragma once

#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>
#include <assert.h>
#include <fstream>
#include "utils.h"

namespace MicroTorch
{

    class BaseModel
    {
    public:
        BaseModel(): m_normMean(0.f), m_normStd(1.f) {}
        BaseModel(float norm_mean, float norm_std): m_normMean(norm_mean), m_normStd(norm_std) { assert( norm_std > 0.f); }
        virtual ~BaseModel() = default;

        virtual inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept = 0;
        virtual void loadStateDict(std::map<std::string, nlohmann::json> state_dict) = 0;

        inline void normalise( Eigen::Ref<RowMatrixXf> x ) noexcept
        {
            if(m_normMean != 0.f)
                x.array() -= m_normMean; // center
            if(m_normStd != 1.f)
                x.array() /= m_normStd; // scale
        }

        float getNormMean() const { return m_normMean; }
        float getNormStd() const { return m_normStd; }

        void setNormMean( float value ) { m_normMean = value; }
        void setNormStd( float value ) { assert( value > 0.f ); m_normStd = value; }

    private:
        float m_normMean, m_normStd;
    };

}