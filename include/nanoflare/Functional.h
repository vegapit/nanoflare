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
    };
}