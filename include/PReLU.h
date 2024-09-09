#pragma once

#include <eigen3/Eigen/Dense>
#include <assert.h>
#include "utils.h"

namespace MicroTorch
{
    class PReLU
    {
    public:
        PReLU(size_t num_channels) : m_numChannels(num_channels), m_w(Eigen::RowVectorXf::Zero(num_channels)) {}
        ~PReLU() = default;

        void setWeight(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_numChannels);
            m_w = v;
        }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {
            RowMatrixXf scaled_neg = x.cwiseMin(0.f).array().colwise() * m_w.transpose().eval().array();
            return x.cwiseMax(0.f) + scaled_neg;
        }

        size_t getNumChannels() const { return m_numChannels; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadVector( std::string("weight"), state_dict );
            setWeight( w );
        }

    private:
        Eigen::RowVectorXf m_w;
        size_t m_numChannels;
    };
}