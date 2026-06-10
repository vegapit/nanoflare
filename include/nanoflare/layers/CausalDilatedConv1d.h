#pragma once

#include <Eigen/Dense>
#include <cassert>
#include "nanoflare/utils.h"

namespace Nanoflare
{

    class CausalDilatedConv1d
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CausalDilatedConv1d(size_t in_channels, size_t out_channels, size_t kernel_size, bool bias, size_t dilation) :
            m_inChannels(in_channels), m_outChannels(out_channels),
            m_kernelSize(kernel_size), m_bias(bias), m_dilation(dilation),
            m_wFused(RowMatrixXf::Zero(out_channels, in_channels * kernel_size)),
            m_b(Eigen::VectorXf::Zero(out_channels))
        {}
        ~CausalDilatedConv1d() = default;

        inline void forward(const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y) noexcept
        {
            assert(x.rows() == m_inChannels && "CausalDilatedConv1d.forward: Wrong input shape");
            assert(y.rows() == m_outChannels && y.cols() == x.cols() && "CausalDilatedConv1d.forward: Wrong output shape");

            const int out_len = x.cols();
            if (m_im2col.rows() != (int)(m_inChannels * m_kernelSize) || m_im2col.cols() != out_len)
                m_im2col.resize(m_inChannels * m_kernelSize, out_len);

            buildIm2col(x, out_len);

            y.noalias() = m_wFused * m_im2col;
            if (m_bias)
                y.colwise() += m_b;
        }

        size_t getInChannels()  const { return m_inChannels; }
        size_t getOutChannels() const { return m_outChannels; }
        size_t getKernelSize()  const { return m_kernelSize; }
        size_t getDilation()    const { return m_dilation; }
        bool   useBias()        const { return m_bias; }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadTensor(std::string("weight"), state_dict);
            auto b = loadVector(std::string("bias"), state_dict);
            for (size_t i = 0; i < m_outChannels; i++)
                setWeight(i, w[i]);
            setBias(b);
        }

    private:
        // im2col layout: row j*ks+k holds the time-shifted x.row(j) for kernel tap k.
        // Causal zero-padding: left_pad = dilation*(kernel_size-1) implicit zeros prepended.
        inline void buildIm2col(const Eigen::Ref<const RowMatrixXf>& x, int out_len) noexcept
        {
            const int left_pad = (int)m_dilation * ((int)m_kernelSize - 1);
            m_im2col.setZero();
            for (int j = 0; j < (int)m_inChannels; ++j) {
                for (int k = 0; k < (int)m_kernelSize; ++k) {
                    const int src_offset = k * (int)m_dilation - left_pad; // always <= 0
                    const int t_start = -src_offset;                        // first valid output sample
                    const int len = out_len - t_start;
                    if (len > 0)
                        m_im2col.row(j * m_kernelSize + k).tail(len).noalias() = x.row(j).head(len);
                }
            }
        }

        void setWeight(size_t i, const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_inChannels && m.cols() == m_kernelSize && i < m_outChannels);
            m_wFused.row(i) = Eigen::Map<const Eigen::RowVectorXf>(m.data(), m_inChannels * m_kernelSize);
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v.transpose();
        }

        size_t m_inChannels, m_outChannels, m_kernelSize, m_dilation;
        bool m_bias;
        RowMatrixXf     m_wFused;  // (out_ch, in_ch * kernel_size)
        Eigen::VectorXf m_b;
        RowMatrixXf     m_im2col;  // (in_ch * kernel_size, out_len), lazily resized
    };

}
