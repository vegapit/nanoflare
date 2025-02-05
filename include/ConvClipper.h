#pragma once

#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "utils.h"

namespace MicroTorch
{

    class ConvClipper
    {
    public:
        ConvClipper(size_t input_size, size_t output_size, size_t kernel_size, size_t dilation) : m_conv( input_size, output_size, kernel_size, true, dilation ) {}
        ~ConvClipper() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            auto y = m_conv.forward( x );
            y.array() += (m_coefSoftsign * y).array() / (1.f + (m_coefSoftsign * y).array().abs());
            y.array() += (m_coefTanh * y).array().tanh();
            return y.cwiseMin( m_ceiling ).cwiseMax( m_floor );
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto conv_state_dict = state_dict[std::string("conv")].get<std::map<std::string, nlohmann::json>>();
            m_conv.loadStateDict( conv_state_dict );
            auto floor = loadVector( std::string("floor"), state_dict );
            m_floor = -sigmoid(5.f * floor(0));
            auto ceiling = loadVector( std::string("ceiling"), state_dict );
            m_ceiling = sigmoid(5.f * ceiling(0));
            auto coef_softsign = loadVector( std::string("coef_softsign"), state_dict );
            m_coefSoftsign = coef_softsign(0); 
            auto coef_tanh = loadVector( std::string("coef_tanh"), state_dict );
            m_coefTanh = coef_tanh(0);
        }

    private:

       inline static float sigmoid(const float& x) { return 1.f / (1.f + std::exp(-x)); }

        CausalDilatedConv1d m_conv;
        float m_floor, m_ceiling, m_coefSoftsign, m_coefTanh;
    };
}