#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"

namespace MicroTorch
{

    class CausalDilatedConv1d
    {
    public:
        CausalDilatedConv1d(int in_channels, int out_channels, int kernel_size, bool bias, int dilation) : m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias), m_dilation(dilation), m_internalPadding((dilation * (kernel_size - 1)) / 2), m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(int i = 0; i < out_channels; i++)
            {
                m_w.push_back( RowMatrixXf::Zero(in_channels, kernel_size) );
                m_dilatedW.push_back( RowMatrixXf::Zero(in_channels, dilation * (kernel_size - 1) + 1) );
            }
        }
        ~CausalDilatedConv1d() = default;

        void setWeight(int channel, const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_inChannels);
            assert(m.cols() == m_kernelSize);
            m_w[channel] = m;

            // Calculate dilated weights
            for(int i = 0; i < m.rows(); i++)
            {
                RowMatrixXf row = m.row(i);
                m_dilatedW[channel].row(i) = dilate( row, m_dilation );
            }
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v;
        }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {    
            RowMatrixXf y = RowMatrixXf::Zero(m_outChannels, x.cols());
            for(int i = 0; i < m_outChannels; i++)
            {
                for(int j = 0; j < m_inChannels; j++)
                    if( m_internalPadding > 0)
                    {
                        RowMatrixXf row = x.row(j);
                        RowMatrixXf padded_row = pad(row, m_internalPadding);
                        RowMatrixXf weights = m_dilatedW[i].row(j);
                        y.row(i) += convolve1d( padded_row, weights );
                    }
                    else
                    {
                        RowMatrixXf row = x.row(j); 
                        RowMatrixXf weights = m_dilatedW[i].row(j);
                        y.row(i) += convolve1d( row, weights );
                    }
                if( m_bias )
                    y.row(i).array() += m_b(i);
            }
            return y;
        }
 
        int getInChannels() const { return m_inChannels; }
        int getOutChannels() const { return m_outChannels; }
        int getKernelSize() const { return m_kernelSize; }
        int getBias() const { return m_bias; }
        int getDilation() const { return m_dilation; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadTensor( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            for(int i = 0; i < m_outChannels; i++)
                setWeight( i, w[i] );
            setBias( b );
        }

    private:
        int m_inChannels, m_outChannels, m_kernelSize, m_dilation, m_internalPadding;
        bool m_bias;

        std::vector<RowMatrixXf> m_w, m_dilatedW; // W = [Outs, Ins, Kernel]
        Eigen::RowVectorXf m_b; // B = [Outs]
    };

}