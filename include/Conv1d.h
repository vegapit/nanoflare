#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"
#include <iostream>

namespace MicroTorch
{

    class Conv1d
    {
    public:
        Conv1d(int in_channels, int out_channels, int kernel_size, bool bias) : m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias),m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(int i = 0; i < out_channels; i++)
                m_w.push_back( RowMatrixXf::Zero(in_channels, kernel_size) );
        }
        ~Conv1d() = default;

        void setWeight(int channel, const Eigen::Ref<RowMatrixXf>& m)
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

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {
            float f_bias = static_cast<float>(m_bias);
            RowMatrixXf y = RowMatrixXf::Zero(m_outChannels, x.cols());
            for(int i = 0; i < m_outChannels; i++)
            {
                for(int j = 0; j < m_inChannels; j++)
                    y.row(i) += convolve1d(x.row(j), m_w[i].row(j));
                y.row(i).array() += f_bias * m_b(i);
            }
            return y;
        }

        int getInChannels() const { return m_inChannels; }
        int getOutChannels() const { return m_outChannels; }
        int getKernelSize() const { return m_kernelSize; }
        int getBias() const { return m_bias; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadTensor( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            for(int i = 0; i < m_outChannels; i++)
                setWeight( i, w[i] );
            setBias( b );
        }

    private:
        int m_inChannels, m_outChannels, m_kernelSize;
        bool m_bias;

        std::vector<RowMatrixXf> m_w; // W = [Outs, Ins, Kernel]
        Eigen::RowVectorXf m_b; // B = [Outs]
    };

}