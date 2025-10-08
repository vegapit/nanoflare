#pragma once

#include <Eigen/Dense>
#include <assert.h>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class Linear
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Linear(size_t in_channels, size_t out_channels, bool bias) : m_inChannels(in_channels), m_outChannels(out_channels), m_bias(bias),
            m_w(RowMatrixXf::Zero(out_channels,in_channels)),
            m_transW(RowMatrixXf::Zero(in_channels,out_channels)),
            m_b(Eigen::RowVectorXf::Zero(out_channels))
        {}
        ~Linear() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<const RowMatrixXf>& x ) const noexcept
        {
            if (m_y.rows() != x.rows() || m_y.cols() != m_transW.cols())
                m_y.resize(x.rows(), m_transW.cols());
            
            // Matrix multiplication 
            m_y.noalias() = x * m_transW;
            
            if( m_bias )
                m_y.rowwise() += m_b;
            return m_y;
        }

        inline RowMatrixXf forwardTranspose(const Eigen::Ref<const RowMatrixXf>& x) const noexcept
        {
            // ensure scratch buffer
            if (m_y.rows() != m_outChannels || m_y.cols() != x.cols())
                m_y.resize(m_outChannels, x.cols());

            // multiply directly: W * x
            // m_w is [out, in], x is [in, time] if channels==in
            m_y.noalias() = m_w * x;

            if (m_bias)
                m_y.colwise() += m_b.transpose(); // broadcast bias across time
            return m_y;
        }

        size_t getInChannels() const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        bool useBias() const { return m_bias; }
        
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

        mutable RowMatrixXf m_y;
        RowMatrixXf m_w, m_transW;
        Eigen::RowVectorXf m_b;

        size_t m_inChannels, m_outChannels;
        bool m_bias;
    };
}