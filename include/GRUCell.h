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
        GRUCell(int inputSize, int hiddenSize, bool bias) : m_hiddenSize(hiddenSize), m_inputSize(inputSize), m_bias(bias) {}
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
            assert(m.cols() == m_inputSize);
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
        bool getBias() const { return m_bias; }

        void forward( const Eigen::Ref<Eigen::RowVectorXf>& x, Eigen::Ref<Eigen::VectorXf> h ) const
        {
            float (*sigmoidPtr)(float) { &sigmoid };
            float (*tanhPtr)(float) { &tanh };

            RowMatrixXf r = m_wih.middleRows(0,m_hiddenSize) * x + m_whh.middleRows(0,m_hiddenSize) * h;
            RowMatrixXf z = m_wih.middleRows(m_hiddenSize,m_hiddenSize) * x + m_whh.middleRows(m_hiddenSize,m_hiddenSize) * h;
            RowMatrixXf nx = m_wih.middleRows(2*m_hiddenSize,m_hiddenSize) * x;
            RowMatrixXf nh = m_whh.middleRows(2*m_hiddenSize,m_hiddenSize) * h;
                
            if(m_bias)
            {
                r += m_bih.segment(0,m_hiddenSize) + m_bhh.segment(0,m_hiddenSize);
                z += m_bih.segment(m_hiddenSize,m_hiddenSize) + m_bhh.segment(m_hiddenSize,m_hiddenSize);
                nx += m_bih.segment(2*m_hiddenSize,m_hiddenSize);
                nh += m_bhh.segment(2*m_hiddenSize,m_hiddenSize);
            }

            z = z.unaryExpr(sigmoidPtr);
            RowMatrixXf n = (nx + r.unaryExpr(sigmoidPtr).cwiseProduct(nh)).unaryExpr(tanhPtr);
            h = (Eigen::MatrixXf::Ones(1,m_hiddenSize) - z).cwiseProduct(n) + z.cwiseProduct(h);
        }

    private:

        int m_inputSize, m_hiddenSize;
        bool m_bias;

        RowMatrixXf m_wih, m_whh;
        Eigen::VectorXf m_bih, m_bhh;
    };

}