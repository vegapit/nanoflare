#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"

namespace MicroTorch
{

    class CausalDilatedConv1d
    {
    public:
        CausalDilatedConv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias, size_t dilation) : m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias), m_dilation(dilation), m_internalPadding(dilation * (kernel_size - 1)), 
            m_w(std::vector<RowMatrixXf>(out_channels)), m_dilatedW(std::vector<RowMatrixXf>(out_channels)),  m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(auto i = 0; i < out_channels; i++)
            {
                m_w[i] = RowMatrixXf::Zero(in_channels, kernel_size);
                m_dilatedW[i] = RowMatrixXf::Zero(in_channels, dilation * (kernel_size - 1) + 1);
            }
        }
        ~CausalDilatedConv1d() = default;

        void setWeight(size_t channel, const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_inChannels);
            assert(m.cols() == m_kernelSize);
            m_w[channel] = m;

            // Calculate dilated weights
            Eigen::RowVectorXf weight_row( m.cols() );
            for(auto i = 0; i < m.rows(); i++)
            {
                weight_row = m.row(i);
                m_dilatedW[channel].row(i) = dilate( weight_row, m_dilation );
            }
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v;
        }

        inline size_t getOutputLength( size_t in_length ) const { return in_length; }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {   
            auto seqLength = x.cols();

            // Build padded input matrix
            RowMatrixXf padded_x( x.rows(), seqLength + m_internalPadding);

            std::vector<Eigen::RowVectorXf> x_rows(x.rowwise().begin(), x.rowwise().end());
            for(auto i = 0; i < x.rows(); i++)
                padded_x.row(i) = padLeft(x_rows[i], m_internalPadding);

            std::vector<Eigen::RowVectorXf> padded_x_rows( padded_x.rowwise().begin(), padded_x.rowwise().end() );

            RowMatrixXf y = RowMatrixXf::Zero(m_outChannels, seqLength);
            for(auto i = 0; i < m_outChannels; i++)
            {
                std::vector<Eigen::RowVectorXf> dilated_weight_rows( m_dilatedW[i].rowwise().begin(), m_dilatedW[i].rowwise().end() );
                for(auto j = 0; j < m_inChannels; j++)
                    y.row(i) += convolve1d( padded_x_rows[j], dilated_weight_rows[j] );
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
        size_t m_inChannels, m_outChannels, m_kernelSize, m_dilation, m_internalPadding;
        bool m_bias;

        std::vector<RowMatrixXf> m_w, m_dilatedW; // W = [Outs, Ins, Kernel]
        Eigen::RowVectorXf m_b; // B = [Outs]
    };

}