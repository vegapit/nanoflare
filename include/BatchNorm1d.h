#pragma once

#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"

namespace MicroTorch
{

    class BatchNorm1d
    {
    public:
        BatchNorm1d(size_t num_channels) : 
            m_numChannels(num_channels), m_w(Eigen::RowVectorXf::Ones(num_channels)), m_b(Eigen::RowVectorXf::Zero(num_channels)),
            m_runningMean(Eigen::RowVectorXf::Zero(num_channels)), m_runningVar(Eigen::RowVectorXf::Ones(num_channels))
        {}
        ~BatchNorm1d() = default;

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

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {
            RowMatrixXf y(x);
            y.array().colwise() -= m_runningMean.transpose().eval().array();
            y.array().colwise() /= (m_runningVar.transpose().eval().array() + 1e-5).sqrt();

            y.array().colwise() *= m_w.transpose().eval().array();
            y.array().colwise() += m_b.transpose().eval().array();

            return y;
        }

        size_t getNumChannels() const { return m_numChannels; }
        
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
        }

    private:
        size_t m_numChannels;
        Eigen::RowVectorXf m_w, m_b, m_runningMean, m_runningVar;
    };

}