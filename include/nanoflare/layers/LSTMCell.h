#pragma once

#include <Eigen/Dense>
#include <cassert>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class LSTMCell
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LSTMCell(size_t input_size, size_t hidden_size, bool bias) :
            m_hiddenSize(hidden_size), m_inputSize(input_size), m_bias(bias),
            m_wCombined(Eigen::MatrixXf::Zero(4*hidden_size, input_size + hidden_size + 1)),
            m_extXH(Eigen::VectorXf::Zero(input_size + hidden_size + 1)),
            m_gates(Eigen::VectorXf::Zero(4*hidden_size)),
            m_cNew(Eigen::VectorXf::Zero(hidden_size)),
            m_bih(Eigen::VectorXf::Zero(4*hidden_size)),
            m_bhh(Eigen::VectorXf::Zero(4*hidden_size))
        {
            m_extXH(input_size + hidden_size) = 1.0f;
        }
        ~LSTMCell() = default;

        void setWeightIH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_hiddenSize && m.cols() == m_inputSize);
            m_wCombined.leftCols(m_inputSize) = m;
        }

        void setWeightHH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_hiddenSize && m.cols() == m_hiddenSize);
            m_wCombined.middleCols(m_inputSize, m_hiddenSize) = m;
        }

        void setBiasIH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 4 * m_hiddenSize);
            m_bih = v;
            m_wCombined.col(m_inputSize + m_hiddenSize) = m_bih + m_bhh;
        }

        void setBiasHH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 4 * m_hiddenSize);
            m_bhh = v;
            m_wCombined.col(m_inputSize + m_hiddenSize) = m_bih + m_bhh;
        }

        size_t getInputSize()  const { return m_inputSize; }
        size_t getHiddenSize() const { return m_hiddenSize; }
        bool   isBiased()      const { return m_bias; }

        inline void forward(const Eigen::Ref<const Eigen::VectorXf>& x,
                            Eigen::Ref<Eigen::VectorXf> h,
                            Eigen::Ref<Eigen::VectorXf> c) noexcept
        {
            m_extXH.head(m_inputSize)                          = x;
            m_extXH.segment(m_inputSize, m_hiddenSize)         = h;
            // trailing 1 is set once in constructor and never changes

            m_gates.noalias() = m_wCombined * m_extXH;

            auto i_gate = m_gates.head(m_hiddenSize);
            auto f_gate = m_gates.segment(m_hiddenSize, m_hiddenSize);
            auto g_gate = m_gates.segment(2 * m_hiddenSize, m_hiddenSize);
            auto o_gate = m_gates.tail(m_hiddenSize);

            m_cNew.array() = f_gate.array().logistic() * c.array()
                           + i_gate.array().logistic() * g_gate.array().tanh();
            c = m_cNew;
            h = o_gate.array().logistic() * m_cNew.array().tanh();
        }

    private:
        size_t m_inputSize, m_hiddenSize;
        bool m_bias;
        Eigen::MatrixXf m_wCombined;  // [W_ih | W_hh | (b_ih+b_hh)], shape (4H, in+H+1)
        Eigen::VectorXf m_extXH;      // [x; h; 1], trailing 1 fixed at construction
        Eigen::VectorXf m_gates, m_cNew;
        Eigen::VectorXf m_bih, m_bhh; // kept to correctly fuse when set independently
    };
}
