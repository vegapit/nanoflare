#pragma once

#include <Eigen/Dense>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class Functional
    {
    public:

        static inline void ReLU( Eigen::Ref<RowMatrixXf> x ) noexcept
        {
            x = x.cwiseMax( 0.f );
        }

        static inline void LeakyReLU( Eigen::Ref<RowMatrixXf> x, float negative_slope) noexcept
        {
            auto a = x.array();
            a = (a > 0).select(a, a * negative_slope);
        }
        
        static inline void Sigmoid( Eigen::Ref<RowMatrixXf> x ) noexcept
        {
            x.array() = x.array().logistic();
        }

        static inline void Tanh( Eigen::Ref<RowMatrixXf> x ) noexcept
        {
            x.array() = x.array().tanh();
        }

        static inline void LayerNorm( Eigen::Ref<RowMatrixXf> x ) noexcept
        {
            // Compute mean and variance per column
            Eigen::RowVectorXf m1 = x.colwise().mean();
            Eigen::RowVectorXf m2 = x.array().square().colwise().mean();
            Eigen::RowVectorXf var = m2.array() - m1.array().square();

            // Inverse std per column
            Eigen::RowVectorXf inv_std = (var.array() + 1e-5f).rsqrt();

            // Normalize in place: (x - mean) / std
            x = (x.rowwise() - m1).array().rowwise() * inv_std.array();
        }
    };
}