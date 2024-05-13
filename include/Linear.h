#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"

namespace MicroTorch
{
    class Linear
    {
    public:
        Linear(int in_channels, int out_channels, bool bias) : m_inChannels(in_channels), m_outChannels(out_channels), m_bias(bias) {}
        ~Linear() = default;

        void setWeight(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_outChannels);
            assert(m.cols() == m_inChannels);
            m_w = m;
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v;
        }

        RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const
        {
            assert(x.cols() == m_inChannels);
            RowMatrixXf y( x.rows(), m_outChannels );
            for(int i = 0; i < x.rows(); i++)
                y.row(i) = x.row(i) * m_w.transpose();
            if(m_bias)
                y.rowwise() += m_b;
            return y;
        }

        int getInChannels() { return m_inChannels; }
        int getOutChannels() { return m_outChannels; }
        int getBias() { return m_bias; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadMatrix( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            setWeight( w );
            setBias( b );
        }

    private:
        RowMatrixXf m_w;
        Eigen::RowVectorXf m_b;

        int m_inChannels, m_outChannels;
        bool m_bias;
    };
}