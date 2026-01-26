#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class Biquad
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Biquad() : m_b0(1.0f), m_b1(0.0f), m_b2(0.0f), m_a1(0.0f), m_a2(0.0f) {}
        ~Biquad() = default;

        inline void forward(const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y) noexcept
        {
            assert(x.rows() == y.rows() && x.cols() == y.cols() && "Biquad.forward: Input and output must have same shape");

            const size_t channels = x.rows();
            const size_t samples = x.cols();

            // Apply biquad filter to each channel independently
            // Using Direct Form II Transposed
            for (size_t ch = 0; ch < channels; ++ch)
            {
                float z1 = 0.0f;
                float z2 = 0.0f;

                for (size_t n = 0; n < samples; ++n)
                {
                    float input = x(ch, n);
                    y(ch, n) = m_b0 * input + z1;
                    z1 = m_b1 * input + z2 - m_a1 * y(ch, n);
                    z2 = m_b2 * input - m_a2 * y(ch, n);
                }
            }
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            // Load precomputed biquad coefficients
            state_dict.at("b0").get_to(m_b0);
            state_dict.at("b1").get_to(m_b1);
            state_dict.at("b2").get_to(m_b2);
            state_dict.at("a1").get_to(m_a1);
            state_dict.at("a2").get_to(m_a2);
        }

    private:
        // Biquad coefficients
        float m_b0, m_b1, m_b2;  // Numerator coefficients
        float m_a1, m_a2;         // Denominator coefficients (a0 = 1.0 is implicit)
    };
}
