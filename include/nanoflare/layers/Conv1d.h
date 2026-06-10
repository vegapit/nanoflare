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
            m_wFused(RowMatrixXf::Zero(out_channels, in_channels * kernel_size)),
            m_b(Eigen::VectorXf::Zero(out_channels))
        {}
        ~Conv1d() = default;

        inline size_t getOutputLength(size_t in_length) const { return in_length - (m_kernelSize - 1); }

        inline void forward(const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y) noexcept
        {
            assert(x.rows() == m_inChannels && "Conv1d.forward: Wrong input shape");
            const int out_len = (int)x.cols() - (int)m_kernelSize + 1;
            assert(y.rows() == m_outChannels && y.cols() == out_len && "Conv1d.forward: Wrong output shape");

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
        // im2col layout: row j*ks+k holds x.row(j) shifted by k, length out_len
        inline void buildIm2col(const Eigen::Ref<const RowMatrixXf>& x, int out_len) noexcept
        {
            for (int j = 0; j < (int)m_inChannels; ++j)
                for (int k = 0; k < (int)m_kernelSize; ++k)
                    m_im2col.row(j * m_kernelSize + k).noalias() = x.row(j).segment(k, out_len);
        }

        void setWeight(size_t i, const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == m_inChannels && m.cols() == m_kernelSize && i < m_outChannels);
            // RowMatrixXf memory: [w(j=0,k=0..ks-1) | w(j=1,k=0..ks-1) | ...]
            // matches im2col row layout j*ks+k -> w(j,k)
            m_wFused.row(i) = Eigen::Map<const Eigen::RowVectorXf>(m.data(), m_inChannels * m_kernelSize);
        }

        void setBias(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_outChannels);
            m_b = v.transpose();
        }

        size_t m_inChannels, m_outChannels, m_kernelSize;
        bool m_bias;
        RowMatrixXf     m_wFused;  // (out_ch, in_ch * kernel_size)
        Eigen::VectorXf m_b;
        RowMatrixXf     m_im2col;  // (in_ch * kernel_size, out_len), lazily resized
    };

}
