#pragma once

#include <eigen3/Eigen/Dense>
#include <assert.h>
#include "utils.h"

namespace MicroTorch
{
    class Linear
    {
    public:
        Linear(size_t in_channels, size_t out_channels, bool bias) : m_inChannels(in_channels), m_outChannels(out_channels), m_bias(bias),
            m_w(RowMatrixXf::Zero(out_channels,in_channels)),
            m_transW(RowMatrixXf::Zero(in_channels,out_channels)),
            m_b(Eigen::RowVectorXf::Zero(out_channels))
        {}
        ~Linear() = default;

        void setWeight(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_outChannels);
            assert(m.cols() == m_inChannels);
            m_w = m;
            m_transW = m.transpose();
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v;
        }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {
            RowMatrixXf y = x * m_transW;
            if( m_bias )
                y.rowwise() += m_b;
            return y;
        }

        size_t getInChannels() const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        size_t getBias() const { return m_bias; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadMatrix( std::string("weight"), state_dict );
            setWeight( w );
            if(m_bias)
            {
                auto b = loadVector( std::string("bias"), state_dict );
                setBias( b );
            }
        }

    private:
        RowMatrixXf m_w, m_transW;
        Eigen::RowVectorXf m_b;

        size_t m_inChannels, m_outChannels;
        bool m_bias;
    };
}