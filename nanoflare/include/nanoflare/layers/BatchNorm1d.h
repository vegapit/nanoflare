#pragma once

#include <Eigen/Dense>
#include "nanoflare/utils.h"

namespace NanoFlare
{

    class BatchNorm1d
    {
    public:
        BatchNorm1d(size_t num_channels) : 
            m_numChannels(num_channels), m_w(Eigen::RowVectorXf::Ones(num_channels)), m_b(Eigen::RowVectorXf::Zero(num_channels)),
            m_runningMean(Eigen::RowVectorXf::Zero(num_channels)), m_runningVar(Eigen::RowVectorXf::Ones(num_channels)),
            m_factor(Eigen::RowVectorXf::Ones(num_channels)), m_bias(Eigen::RowVectorXf::Zero(num_channels))
        {}
        ~BatchNorm1d() = default;

        inline void apply( Eigen::Ref<RowMatrixXf> x ) noexcept
        {
            x.array().colwise() *= m_factor.transpose().eval().array();
            x.array().colwise() += m_bias.transpose().eval().array();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadVector( std::string("weight"), state_dict );
            setWeight( w );
            auto b = loadVector( std::string("bias"), state_dict );
            setBias( b );
            auto running_mean = loadVector( std::string("running_mean"), state_dict );
            setRunningMean( running_mean );
            auto running_var = loadVector( std::string("running_var"), state_dict );
            setRunningVar( running_var );

            m_factor.array() = m_w.array() / (m_runningVar.array() + 1e-5).sqrt();
            m_bias.array() = m_b.array() - m_runningMean.array() * m_factor.array();
        }

    private:
        
        void setRunningMean(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_numChannels);
            m_runningMean = v;
        }

        void setRunningVar(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_numChannels);
            m_runningVar= v;
        }

        void setWeight(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_numChannels);
            m_w = v;
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_numChannels);
            m_b = v;
        }

        size_t m_numChannels;
        Eigen::RowVectorXf m_w, m_b, m_runningMean, m_runningVar, m_factor, m_bias;
    };

}