#pragma once

#include <eigen3/Eigen/Dense>
#include "BaseModel.h"
#include "ResidualBlock.h"
#include "CausalDilatedConv1d.h"
#include "Linear.h"
#include "utils.h"

namespace MicroTorch
{

    class WaveNet : public BaseModel
    {
    public:
        WaveNet(size_t input_size, size_t num_channels, size_t output_size, size_t kernel_size, std::vector<size_t> dilations, size_t stack_size, bool gated, Activation activation, float norm_mean, float norm_std) : 
            BaseModel(norm_mean, norm_std), m_numChannels(num_channels), m_dilations(dilations), m_stackSize(stack_size), m_gated(gated), m_activation(activation),
            m_inputConv(input_size, num_channels, kernel_size, true, 1),
            m_outputLinear(num_channels, output_size, false)
        {
            for(size_t k = 0; k < stack_size; k++)
                for(auto dilation: dilations)
                    m_blockStack.push_back( ResidualBlock(num_channels, kernel_size, dilation, true, true, gated, activation) );
        }
        ~WaveNet() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            size_t dilations_size = m_dilations.size();

            RowMatrixXf norm_x( x );
            normalise( norm_x );
            RowMatrixXf y = m_inputConv.forward( norm_x );

            RowMatrixXf skip_sum = RowMatrixXf::Zero( m_numChannels, x.cols() );
            RowMatrixXf skip_y( m_numChannels, x.cols() );
            for(size_t k = 0; k < m_stackSize; k++)
                for(size_t i = 0; i < dilations_size; i++)
                {
                    std::tie( y, skip_y ) = m_blockStack[k * dilations_size + i].forward( y );
                    skip_sum.array() += skip_y.array();
                }
            skip_sum = skip_sum.cwiseMax(0.f); // Apply ReLU
            skip_sum.transposeInPlace(); // Transpose
            return m_outputLinear.forward( skip_sum ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            auto conv_state_dict = state_dict[std::string("conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( conv_state_dict );
            for(size_t k = 0; k < m_stackSize; k++)
                for(size_t i = 0; i < m_dilations.size(); i++)
                {
                    size_t idx = k * m_dilations.size() + i;
                    auto block_state_dict = state_dict[std::string("blockStack.") + std::to_string(idx)].get<std::map<std::string, nlohmann::json>>();
                    m_blockStack[idx].loadStateDict( block_state_dict );
                }
            auto linear_state_dict = state_dict[std::string("linear")].get<std::map<std::string, nlohmann::json>>();
            m_outputLinear.loadStateDict( linear_state_dict );
        }

    private:
        size_t m_numChannels, m_stackSize;
        bool m_gated;
        Activation m_activation;
        std::vector<size_t> m_dilations;
        CausalDilatedConv1d m_inputConv;
        std::vector<ResidualBlock> m_blockStack;
        Linear m_outputLinear;
    };
}