#pragma once

#include <Eigen/Dense>
#include "nanoflare/utils.h"

namespace Nanoflare
{

    class Conv1d
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias) : 
            m_inChannels(in_channels), m_outChannels(out_channels), m_w(std::vector<RowMatrixXf>(out_channels)),
            m_kernelSize(kernel_size), m_bias(bias), m_b(Eigen::RowVectorXf::Zero(out_channels))
        {
            for(auto i = 0; i < out_channels; i++)
                m_w[i] = RowMatrixXf::Zero(in_channels, kernel_size);
        }
        ~Conv1d() = default;

        inline size_t getOutputLength( size_t in_length ) const { return in_length - (m_kernelSize - 1); }

        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, RowMatrixXf& y ) const noexcept
        {
            if(x.data() == y.data())
            {
                RowMatrixXf temp = RowMatrixXf::Zero(m_outChannels, getOutputLength( x.cols() ));
                process( x, temp );
                y = std::move(temp);
            }
            else
            {
                if (y.rows() != m_outChannels || y.cols() != getOutputLength( x.cols() ))
                    y.resize(m_outChannels, getOutputLength( x.cols() ));
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

        inline void process( const Eigen::Ref<const RowMatrixXf>& x, RowMatrixXf& mat ) const noexcept
        {
            if (m_out.size() != mat.cols())
                m_out.resize( mat.cols() );
            for(auto i = 0; i < m_outChannels; i++)
            {
                for(auto j = 0; j < m_inChannels; j++)
                {
                    convolve1d( x.row(j), m_w[i].row(j), m_out );
                    mat.row(i) += m_out;
                }
                if( m_bias )
                    mat.row(i).array() += m_b(i);
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
        mutable RowMatrixXf m_y;
        mutable Eigen::RowVectorXf m_out;
    };

}