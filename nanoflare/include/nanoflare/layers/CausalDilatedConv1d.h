#pragma once

#include <Eigen/Dense>
#include "nanoflare/utils.h"

namespace NanoFlare
{

    class CausalDilatedConv1d
    {
    public:
        CausalDilatedConv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias, size_t dilation) : m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias), m_dilation(dilation), 
            m_w(std::vector<RowMatrixXf>(out_channels)), m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(auto i = 0; i < out_channels; i++)
                m_w[i] = RowMatrixXf::Zero(in_channels, kernel_size);
        }
        ~CausalDilatedConv1d() = default;

        inline size_t getOutputLength( size_t in_length ) const { return in_length; }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {   
            RowMatrixXf y = RowMatrixXf::Zero(m_outChannels, x.cols());
            for(auto i = 0; i < m_outChannels; i++)
            {
                for(auto j = 0; j < m_inChannels; j++)
                    y.row(i) += dilatedcausalconvolve1d( x.row(j), m_w[i].row(j), m_dilation );
                if( m_bias )
                    y.row(i).array() += m_b(i);
            }
            return y;
        }

        size_t getInChannels() const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        size_t getKernelSize() const { return m_kernelSize; }
        size_t getDilation() const { return m_dilation; }
        bool useBias() const { return m_bias; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {   
            auto w = loadTensor( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            for(auto i = 0; i < m_outChannels; i++)
                setWeight( i, w[i] );
            setBias( b );
        }

    private:

        void setWeight(size_t channel, const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_inChannels);
            assert(m.cols() == m_kernelSize);
            m_w[channel] = m;
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v;
        }

        size_t m_inChannels, m_outChannels, m_kernelSize, m_dilation;
        bool m_bias;

        std::vector<RowMatrixXf> m_w; // W = [Outs, Ins, Kernel]
        Eigen::RowVectorXf m_b; // B = [Outs]
    };

}