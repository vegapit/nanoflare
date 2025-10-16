#pragma once

#include <Eigen/Dense>
#include <cassert>
#include "nanoflare/utils.h"

namespace Nanoflare
{

    class Conv1d
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) : 
            m_inChannels(in_channels), m_outChannels(out_channels), 
            m_kernelSize(kernel_size), m_bias(bias), 
            m_w(std::vector<RowMatrixXf>(out_channels)),
            m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(auto i = 0; i < out_channels; i++)
                m_w[i] = RowMatrixXf::Zero(in_channels, kernel_size);
        }
        ~Conv1d() = default;

        inline size_t getOutputLength( size_t in_length ) const { return in_length - (m_kernelSize - 1); }

        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y ) noexcept
        {
            assert(x.rows() == m_inChannels && "Conv1d.forward: Wrong input shape");
            assert((y.rows() == m_outChannels && y.cols() == getOutputLength( x.cols() )) && "Conv1d.forward: Wrong output shape");

            if(x.data() == y.data())
            {
                RowMatrixXf temp = RowMatrixXf::Zero( y.rows(), y.cols() );
                process( x, temp );
                y = std::move( temp );
            }
            else
            {
                y.setZero();
                process( x, y );
            }
        }

        size_t getInChannels() const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        size_t getKernelSize() const { return m_kernelSize; }
        bool useBias() const { return m_bias; }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadTensor( std::string("weight"), state_dict );
            auto b = loadVector( std::string("bias"), state_dict );
            for(size_t i = 0; i < m_outChannels; i++)
                setWeight( i, w[i] );
            setBias( b );
        }

    private:

        inline void process( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> mat ) noexcept
        {           
            const int out_len = x.cols() - m_kernelSize + 1;
            assert(mat.cols() == out_len && mat.rows() == m_outChannels);
            
            // Build im2col matrices for each input channel
            std::vector<Eigen::MatrixXf> i2c_mat;
            i2c_mat.reserve(m_inChannels);
            for (int j = 0; j < m_inChannels; ++j)
                i2c_mat.emplace_back( im2col(x.row(j), m_kernelSize) ); // (out_len, kernel_size)

            for (int i = 0; i < m_outChannels; ++i)
            {
                mat.row(i).setConstant(m_bias ? m_b(i) : 0.f);
                for (int j = 0; j < m_inChannels; ++j)
                    mat.row(i).noalias() += i2c_mat[j] * m_w[i].row(j).transpose();
            }
        }

        void setWeight(size_t channel, const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_inChannels);
            assert(m.cols() == m_kernelSize);
            assert(channel < m_outChannels);
            m_w[channel] = m;
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v;
        }

        size_t m_inChannels, m_outChannels, m_kernelSize;
        bool m_bias;

        std::vector<RowMatrixXf> m_w; // W = [Outs, Ins, Kernel]
        Eigen::RowVectorXf m_b; // B = [Outs]
    };

}