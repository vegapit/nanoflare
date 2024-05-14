#pragma once

#include <eigen3/Eigen/Dense>
#include <assert.h>
#include <functional>
#include "utils.h"

namespace MicroTorch
{
    class LSTMCell
    {
    public:
        LSTMCell(int inputSize, int hiddenSize, bool bias) : m_hiddenSize(hiddenSize), m_inputSize(inputSize), m_bias(bias) {}
        ~LSTMCell() = default;

        void setWeightIH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_inputSize);
            assert(m.cols() == m_hiddenSize);
            m_wih = m;
        }

        void setWeightHH(const Eigen::Ref<RowMatrixXf>& m)
        {
            assert(m.rows() == 4 * m_inputSize);
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

        int getInputSize() const { return m_inputSize; } 
        int getHiddenSize() const { return m_hiddenSize; }
        bool getBias() const { return m_bias; }

        void forward( const Eigen::Ref<Eigen::RowVectorXf>& x, Eigen::Ref<Eigen::VectorXf> h, Eigen::Ref<Eigen::VectorXf> c ) const
        {
            float (*sigmoidPtr)(float) { &sigmoid };
            float (*tanhPtr)(float) { &tanh };

            RowMatrixXf i = m_wih.middleRows(0,m_hiddenSize) * x + m_whh.middleRows(0,m_hiddenSize) * h;
            RowMatrixXf f = m_wih.middleRows(m_hiddenSize,m_hiddenSize) * x + m_whh.middleRows(m_hiddenSize,m_hiddenSize) * h;
            RowMatrixXf g = m_wih.middleRows(2*m_hiddenSize,m_hiddenSize) * x + m_whh.middleRows(2*m_hiddenSize,m_hiddenSize) * h;
            RowMatrixXf o = m_wih.middleRows(3*m_hiddenSize,m_hiddenSize) * x + m_whh.middleRows(3*m_hiddenSize,m_hiddenSize) * h;
                
            if(m_bias)
            {
                i += m_bih.segment(0,m_hiddenSize) + m_bhh.segment(0,m_hiddenSize);
                f += m_bih.segment(m_hiddenSize,m_hiddenSize) + m_bhh.segment(m_hiddenSize,m_hiddenSize);
                g += m_bih.segment(2*m_hiddenSize,m_hiddenSize) + m_bhh.segment(2*m_hiddenSize,m_hiddenSize);
                o += m_bih.segment(3*m_hiddenSize,m_hiddenSize) + m_bhh.segment(3*m_hiddenSize,m_hiddenSize);
            }

            c = f.unaryExpr(sigmoidPtr).cwiseProduct(c) + i.unaryExpr(sigmoidPtr).cwiseProduct( g.unaryExpr(tanhPtr) );
            h = o.unaryExpr(sigmoidPtr).cwiseProduct( c.unaryExpr(tanhPtr) );
        }

    private:

        int m_inputSize, m_hiddenSize;
        bool m_bias;

        RowMatrixXf m_wih, m_whh;
        Eigen::VectorXf m_bih, m_bhh;
    };

}