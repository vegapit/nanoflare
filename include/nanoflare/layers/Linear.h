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
        
        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, RowMatrixXf& y ) const noexcept
        {
            if(x.data() == y.data())
            {
                RowMatrixXf temp(x.rows(), m_transW.cols());
                temp.noalias() = x * m_transW;
                if( m_bias )
                    temp.rowwise() += m_b;
                y = std::move(temp);
            }
            else
            {
                if (y.rows() != x.rows() || y.cols() != m_transW.cols())
                    y.resize(x.rows(), m_transW.cols());
                y.noalias() = x * m_transW;
                if( m_bias )
                    y.rowwise() += m_b;
            }
        }

        inline void forwardTranspose(const Eigen::Ref<const RowMatrixXf>& x, RowMatrixXf& y ) const noexcept
        {
            if(x.data() == y.data())
            {
                RowMatrixXf temp(m_w.rows(), x.cols());
                temp.noalias() = m_w * x;
                if( m_bias )
                    temp.colwise() += m_b.transpose();
                y = std::move(temp);
            }
            else
            {
                if (y.rows() != m_w.rows() || y.cols() != x.cols())
                    y.resize(m_w.rows(), x.cols());
                y.noalias() = m_w * x;
                if (m_bias)
                    y.colwise() += m_b.transpose(); // broadcast bias across time
            }

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

        RowMatrixXf m_w, m_transW;
        Eigen::RowVectorXf m_b;

        size_t m_inChannels, m_outChannels;
        bool m_bias;
    };
}