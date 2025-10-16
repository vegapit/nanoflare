#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class LSTMCell
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LSTMCell(size_t input_size, size_t hidden_size, bool bias) 
            : m_hiddenSize(hidden_size),
            m_inputSize(input_size),
            m_bias(bias),
            m_wih(RowMatrixXf::Zero(4*hidden_size, input_size)),
            m_whh(RowMatrixXf::Zero(4*hidden_size, hidden_size)),
            m_bih(Eigen::VectorXf::Zero(4*hidden_size)),
            m_bhh(Eigen::VectorXf::Zero(4*hidden_size)),
            m_w_fused(RowMatrixXf::Zero(4*hidden_size, input_size + hidden_size)),
            m_xh_fused(Eigen::VectorXf::Zero(input_size + hidden_size)),
            m_gates(Eigen::VectorXf::Zero(4*hidden_size)),
            m_weights_fused(false)
        {
            // Pre-fuse biases if using bias
            if (m_bias)
                m_bias_fused = m_bih + m_bhh;
        }

        void setWeightIH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_hiddenSize);
            assert(m.cols() == m_inputSize);
            m_wih = m;
            m_weights_fused = false;  // Mark for re-fusion
        }

        void setWeightHH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_hiddenSize);
            assert(m.cols() == m_hiddenSize);
            m_whh = m;
            m_weights_fused = false;  // Mark for re-fusion
        }

        void setBiasIH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 4 * m_hiddenSize);
            m_bih = v;
            if (m_bias)
                m_bias_fused = m_bih + m_bhh;
        }

        void setBiasHH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 4 * m_hiddenSize);
            m_bhh = v;
            if (m_bias)
                m_bias_fused = m_bih + m_bhh;
        }

        size_t getInputSize() const { return m_inputSize; } 
        size_t getHiddenSize() const { return m_hiddenSize; }
        bool isBiased() const { return m_bias; }

        inline void forward(const Eigen::Ref<const Eigen::VectorXf>& x, 
                        Eigen::Ref<Eigen::VectorXf> h, 
                        Eigen::Ref<Eigen::VectorXf> c) noexcept 
        {
            // Fuse weights once if not already done
            if (!m_weights_fused)
            {
                m_w_fused << m_wih, m_whh;
                m_weights_fused = true;
            }

            // Fuse inputs (x and h)
            m_xh_fused << x, h;

            // Single matrix-vector multiplication into pre-allocated buffer
            m_gates.noalias() = m_w_fused * m_xh_fused;

            // Add fused bias if needed
            if (m_bias)
                m_gates += m_bias_fused;

            // Extract gate segments (views, no copy)
            auto i_gate = m_gates.head(m_hiddenSize);
            auto f_gate = m_gates.segment(m_hiddenSize, m_hiddenSize);
            auto g_gate = m_gates.segment(2 * m_hiddenSize, m_hiddenSize);
            auto o_gate = m_gates.tail(m_hiddenSize);

            // LSTM computations
            c = f_gate.array().logistic() * c.array() + i_gate.array().logistic() * g_gate.array().tanh();
            h = o_gate.array().logistic() * c.array().tanh();
        }

    private:
        size_t m_inputSize, m_hiddenSize;
        bool m_bias;
        RowMatrixXf m_wih, m_whh, m_w_fused;
        Eigen::VectorXf m_bih, m_bhh, m_xh_fused, m_gates, m_bias_fused;
        bool m_weights_fused;
    };

}