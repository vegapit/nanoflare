#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <assert.h>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class LSTMCell
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LSTMCell(size_t input_size, size_t hidden_size, bool bias) : m_hiddenSize(hidden_size), m_inputSize(input_size), m_bias(bias),
            m_wih(RowMatrixXf::Zero(4*hidden_size,input_size)),
            m_whh(RowMatrixXf::Zero(4*hidden_size,hidden_size)),
            m_bih(Eigen::VectorXf::Zero(4*hidden_size)),
            m_bhh(Eigen::VectorXf::Zero(4*hidden_size))
        {}
        ~LSTMCell() = default;

        void setWeightIH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_hiddenSize);
            assert(m.cols() == m_inputSize);
            m_wih = m;
        }

        void setWeightHH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_hiddenSize);
            assert(m.cols() == m_hiddenSize);
            m_whh = m;
        }

        void setBiasIH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 4 * m_hiddenSize);
            m_bih = v;
        }

        void setBiasHH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 4 * m_hiddenSize);
            m_bhh = v;
        }

        size_t getInputSize() const { return m_inputSize; } 
        size_t getHiddenSize() const { return m_hiddenSize; }
        bool isBiased() const { return m_bias; }

        inline void forward(const Eigen::Ref<const Eigen::VectorXf>& x, Eigen::Ref<Eigen::VectorXf> h, Eigen::Ref<Eigen::VectorXf> c) const noexcept {
            Eigen::VectorXf mat = (m_wih * x) + (m_whh * h); // Key optimization

            auto i_inner = mat.head(m_hiddenSize);
            auto f_inner = mat.segment(m_hiddenSize, m_hiddenSize);
            auto g_inner = mat.segment(2 * m_hiddenSize, m_hiddenSize);
            auto o_inner = mat.tail(m_hiddenSize);

            if (m_bias) {
                i_inner += m_bih.head(m_hiddenSize) + m_bhh.head(m_hiddenSize);
                f_inner += m_bih.segment(m_hiddenSize, m_hiddenSize) + m_bhh.segment(m_hiddenSize, m_hiddenSize);
                g_inner += m_bih.segment(2 * m_hiddenSize, m_hiddenSize) + m_bhh.segment(2 * m_hiddenSize, m_hiddenSize);
                o_inner += m_bih.tail(m_hiddenSize) + m_bhh.tail(m_hiddenSize);
            }

            c.array() = f_inner.array().logistic() * c.array() + i_inner.array().logistic() * g_inner.array().tanh();
            h.array() = o_inner.array().logistic() * c.array().tanh();
        }

    private:
        size_t m_inputSize, m_hiddenSize;
        bool m_bias;

        RowMatrixXf m_wih, m_whh;
        Eigen::VectorXf m_bih, m_bhh;
    };

}