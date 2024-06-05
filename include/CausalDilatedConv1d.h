#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"
#include <iostream>

namespace MicroTorch
{

    class CausalDilatedConv1d
    {
    public:
        CausalDilatedConv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias, size_t dilation) : m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias), m_dilation(dilation), m_internalPadding(dilation * (kernel_size - 1)), m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(size_t i = 0; i < out_channels; i++)
            {
                m_w.push_back( RowMatrixXf::Zero(in_channels, kernel_size) );
                m_dilatedW.push_back( RowMatrixXf::Zero(in_channels, dilation * (kernel_size - 1) + 1) );
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
            for(Eigen::Index i = 0; i < m.rows(); i++)
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

        inline size_t getOutputLength( size_t in_length ) const { return in_length + (m_kernelSize - 1); }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {   
            size_t in_length = x.cols();

            // Build padded input matrix
            RowMatrixXf padded_x(x.rows(), in_length + 2 * m_internalPadding);
            Eigen::RowVectorXf input_row( in_length );
            for(Eigen::Index i = 0; i < x.rows(); i++)
            {
                input_row = x.row(i);
                padded_x.row(i) = pad(input_row, m_internalPadding);
            }
 
            Eigen::RowVectorXf padded_input_row( padded_x.cols() );
            Eigen::RowVectorXf dilated_weight_row( m_dilatedW[0].cols() );

            RowMatrixXf y = RowMatrixXf::Zero(m_outChannels, in_length);
            for(size_t i = 0; i < m_outChannels; i++)
            {
                for(size_t j = 0; j < m_inChannels; j++)
                {
                    padded_input_row = padded_x.row(j);
                    dilated_weight_row = m_dilatedW[i].row(j);
                    y.row(i) += convolve1d( padded_input_row, dilated_weight_row, in_length );
                }
                if( m_bias )
                    y.row(i).array() += m_b(i);
            }
            return y;
        }
 
        size_t getInChannels() const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        size_t getKernelSize() const { return m_kernelSize; }
        size_t getBias() const { return m_bias; }
        size_t getDilation() const { return m_dilation; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadTensor( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            for(size_t i = 0; i < m_outChannels; i++)
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