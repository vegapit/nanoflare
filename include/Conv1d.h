#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"

namespace MicroTorch
{

    class Conv1d
    {
    public:
        Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) : m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias),m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(size_t i = 0; i < out_channels; i++)
                m_w.push_back( RowMatrixXf::Zero(in_channels, kernel_size) );
        }
        ~Conv1d() = default;

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

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {
            RowMatrixXf y = RowMatrixXf::Zero(m_outChannels, x.cols());
            for(size_t i = 0; i < m_outChannels; i++)
            {
                for(size_t j = 0; j < m_inChannels; j++)
                {
                    RowMatrixXf row = x.row(j);
                    RowMatrixXf weights = m_w[i].row(j);
                    y.row(i) += convolve1d(row, weights);
                }
                if(m_bias)
                    y.row(i).array() += m_b(i);
            }
            return y;
        }

        size_t getInChannels() const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        size_t getKernelSize() const { return m_kernelSize; }
        size_t getBias() const { return m_bias; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadTensor( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            for(size_t i = 0; i < m_outChannels; i++)
                setWeight( i, w[i] );
            setBias( b );
        }

    private:
        size_t m_inChannels, m_outChannels, m_kernelSize;
        bool m_bias;

        std::vector<RowMatrixXf> m_w; // W = [Outs, Ins, Kernel]
        Eigen::RowVectorXf m_b; // B = [Outs]
    };

}