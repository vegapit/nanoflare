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
            m_numChannels(num_channels), m_w(Eigen::RowVectorXf::Zero(num_channels)), m_b(Eigen::RowVectorXf::Zero(num_channels))
        {}
        ~BatchNorm1d() = default;

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
            auto mean_x = x.rowwise().mean();
            auto mean_x2 = x.array().square().rowwise().mean();
            auto var = mean_x2.array() - mean_x.array().square() + 1e-5;

            RowMatrixXf y(x);
            y.array().colwise() -= mean_x.array();
            y.array().colwise() /= var.array().sqrt();

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
        }

    private:
        size_t m_numChannels;
        Eigen::RowVectorXf m_w, m_b;
    };

}