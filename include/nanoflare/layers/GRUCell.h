#pragma once

#include <Eigen/Dense>
#include <cassert>
#include "nanoflare/Functional.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class GRUCell
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        GRUCell(size_t input_size, size_t hidden_size, bool bias) :
            m_hiddenSize(hidden_size), m_inputSize(input_size), m_bias(bias),
            m_wCombined(Eigen::MatrixXf::Zero(3*hidden_size, input_size+1)),
            m_uCombined(Eigen::MatrixXf::Zero(3*hidden_size, hidden_size+1)),
            m_extX(Eigen::VectorXf::Zero(input_size+1)),
            m_extH(Eigen::VectorXf::Zero(hidden_size+1)),
            m_alpha(Eigen::VectorXf::Zero(3*hidden_size)),
            m_beta(Eigen::VectorXf::Zero(3*hidden_size)),
            m_r(Eigen::VectorXf::Zero(hidden_size)),
            m_z(Eigen::VectorXf::Zero(hidden_size)),
            m_n(Eigen::VectorXf::Zero(hidden_size))
        {
            m_extX(input_size)  = 1.0f;
            m_extH(hidden_size) = 1.0f;
        }
        ~GRUCell() = default;

        void setWeightIH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 3 * m_hiddenSize && m.cols() == m_inputSize);
            m_wCombined.leftCols(m_inputSize) = m;
        }

        void setWeightHH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 3 * m_hiddenSize && m.cols() == m_hiddenSize);
            m_uCombined.leftCols(m_hiddenSize) = m;
        }

        void setBiasIH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 3 * m_hiddenSize);
            m_wCombined.col(m_inputSize) = v;
        }

        void setBiasHH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 3 * m_hiddenSize);
            m_uCombined.col(m_hiddenSize) = v;
        }

        size_t getInputSize()  const { return m_inputSize; }
        size_t getHiddenSize() const { return m_hiddenSize; }
        bool   isBiased()      const { return m_bias; }

        inline void forward(const Eigen::Ref<const Eigen::VectorXf>& x, Eigen::Ref<Eigen::VectorXf> h) noexcept
        {
            m_extX.head(m_inputSize)  = x;
            m_extH.head(m_hiddenSize) = h;

            m_alpha.noalias() = m_wCombined * m_extX;
            m_beta.noalias()  = m_uCombined * m_extH;

            m_r.noalias() = m_alpha.head(m_hiddenSize)                    + m_beta.head(m_hiddenSize);
            m_z.noalias() = m_alpha.segment(m_hiddenSize, m_hiddenSize)   + m_beta.segment(m_hiddenSize, m_hiddenSize);
            m_n.noalias() = m_alpha.tail(m_hiddenSize);

            Functional::Sigmoid(m_r);
            Functional::Sigmoid(m_z);
            m_n.array() += m_r.array() * m_beta.tail(m_hiddenSize).array();
            m_n = m_n.array().tanh();

            // reuse m_r as scratch to avoid aliasing on h
            m_r.array() = (1.f - m_z.array()) * m_n.array() + m_z.array() * h.array();
            h = m_r;
        }

    private:
        size_t m_inputSize, m_hiddenSize;
        bool m_bias;
        Eigen::MatrixXf m_wCombined; // [W_ih | b_ih], shape (3H, in+1)
        Eigen::MatrixXf m_uCombined; // [W_hh | b_hh], shape (3H, H+1)
        Eigen::VectorXf m_extX, m_extH;               // extended input/hidden with trailing 1
        Eigen::VectorXf m_alpha, m_beta, m_r, m_z, m_n; // pre-allocated scratch
    };
}
