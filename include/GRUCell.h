#pragma once

#include <eigen3/Eigen/Dense>
#include <assert.h>
#include <functional>
#include "utils.h"

namespace MicroTorch
{
    class GRUCell
    {
    public:
        GRUCell(int input_size, int hidden_size, bool bias) : m_hiddenSize(hidden_size), m_inputSize(input_size), m_bias(bias),
            m_wih(RowMatrixXf::Zero(3*hidden_size,input_size)),
            m_whh(RowMatrixXf::Zero(3*hidden_size,hidden_size)),
            m_bih(Eigen::VectorXf::Zero(3*hidden_size)),
            m_bhh(Eigen::VectorXf::Zero(3*hidden_size))
        {}
        ~GRUCell() = default;

        void setWeightIH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 3 * m_hiddenSize);
            assert(m.cols() == m_inputSize);
            m_wih = m;
        }

        void setWeightHH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 3 * m_hiddenSize);
            assert(m.cols() == m_hiddenSize);
            m_whh = m;
        }

        void setBiasIH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 3 * m_hiddenSize);
            m_bih = v;
        }

        void setBiasHH(const Eigen::Ref<Eigen::VectorXf>& v)
        {
            assert(v.size() == 3 * m_hiddenSize);
            m_bhh = v;
        }

        int getInputSize() const { return m_inputSize; } 
        int getHiddenSize() const { return m_hiddenSize; }
        bool isBiased() const { return m_bias; }

        inline void forward( const Eigen::Ref<Eigen::VectorXf>& x, Eigen::Ref<Eigen::VectorXf> h ) const noexcept
        {
            Eigen::VectorXf r_inner = m_wih.middleRows(0,m_hiddenSize) * x + m_whh.middleRows(0,m_hiddenSize) * h;
            Eigen::VectorXf z_inner = m_wih.middleRows(m_hiddenSize,m_hiddenSize) * x + m_whh.middleRows(m_hiddenSize,m_hiddenSize) * h;
            Eigen::VectorXf nx_inner = m_wih.middleRows(2*m_hiddenSize,m_hiddenSize) * x;
            Eigen::VectorXf nh_inner = m_whh.middleRows(2*m_hiddenSize,m_hiddenSize) * h;

            if(m_bias)
            {
                r_inner += m_bih.segment(0,m_hiddenSize) + m_bhh.segment(0,m_hiddenSize);
                z_inner += m_bih.segment(m_hiddenSize,m_hiddenSize) + m_bhh.segment(m_hiddenSize,m_hiddenSize);
                nx_inner += m_bih.segment(2*m_hiddenSize,m_hiddenSize);
                nh_inner += m_bhh.segment(2*m_hiddenSize,m_hiddenSize);
            }

            Eigen::VectorXf n_inner;
            n_inner.array() = nx_inner.array() + r_inner.array().logistic() * nh_inner.array();

            Eigen::VectorXf z;
            z.array() = z_inner.array().logistic();

            h.array() = (1.f - z.array()) * n_inner.array().tanh() + z.array() * h.array();
        }

    private:

        int m_inputSize, m_hiddenSize;
        bool m_bias;

        RowMatrixXf m_wih, m_whh;
        Eigen::VectorXf m_bih, m_bhh;
    };

}