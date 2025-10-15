#pragma once

#include <Eigen/Dense>
#include <assert.h>
#include <functional>
#include "nanoflare/Functional.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class GRUCell
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        GRUCell(size_t input_size, size_t hidden_size, bool bias) : m_hiddenSize(hidden_size), m_inputSize(input_size), m_bias(bias),
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

        size_t getInputSize() const { return m_inputSize; } 
        size_t getHiddenSize() const { return m_hiddenSize; }
        bool isBiased() const { return m_bias; }

        inline void forward( const Eigen::Ref<const Eigen::VectorXf>& x, Eigen::Ref<Eigen::VectorXf> h ) const noexcept
        {
            Eigen::VectorXf mat1 = m_wih * x;
            Eigen::VectorXf mat2 = m_whh * h;
            
            auto r_inner = (mat1.head(m_hiddenSize) + mat2.head(m_hiddenSize)).eval();
            auto z_inner = (mat1.segment(m_hiddenSize,m_hiddenSize) + mat2.segment(m_hiddenSize,m_hiddenSize)).eval();
            auto nx_inner = mat1.tail(m_hiddenSize);
            auto nh_inner = mat2.tail(m_hiddenSize);

            if(m_bias) {
                r_inner += m_bih.head(m_hiddenSize) + m_bhh.head(m_hiddenSize);
                z_inner += m_bih.segment(m_hiddenSize,m_hiddenSize) + m_bhh.segment(m_hiddenSize,m_hiddenSize);
                nx_inner += m_bih.tail(m_hiddenSize);
                nh_inner += m_bhh.tail(m_hiddenSize);
            }
            
            auto n_inner = nx_inner.array() + r_inner.array().logistic() * nh_inner.array();    
            Functional::Sigmoid( z_inner );
            h.array() = (1.f - z_inner.array()) * n_inner.array().tanh() + z_inner.array() * h.array();
        }

    private:

        size_t m_inputSize, m_hiddenSize;
        bool m_bias;
        RowMatrixXf m_wih, m_whh;
        Eigen::VectorXf m_bih, m_bhh;
    };

}