#pragma once

#include <eigen3/Eigen/Dense>
#include "BaseModel.h"
#include "ResidualBlock.h"
#include "CausalDilatedConv1d.h"
#include "Linear.h"

namespace MicroTorch
{

    class WaveNet : public BaseModel
    {
    public:
        WaveNet(int input_size, int num_channels, int output_size, int kernel_size, std::vector<int> dilations, int stack_size, float norm_mean, float norm_std) : 
            BaseModel(norm_mean, norm_std), m_numChannels(num_channels), m_dilations(dilations), m_stackSize(stack_size),
            m_inputConv(input_size, num_channels, kernel_size, true, 1),
            m_outputLinear(num_channels * dilations.size() * stack_size, output_size, true)
        {
            for(int k = 0; k < stack_size; k++)
                for(auto dilation: dilations)
                    m_blockStack.push_back( ResidualBlock(num_channels, kernel_size, dilation, true, true, Activation::TANH, Activation::SIGMOID) );
        }
        ~WaveNet() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override
        {
            RowMatrixXf norm_x = x;
            normalise( norm_x );
            RowMatrixXf y = m_inputConv.forward( norm_x );

            RowMatrixXf skip_ys( m_numChannels * m_dilations.size(), x.cols() );
            RowMatrixXf skip_y;
            for(int k = 0; k < m_stackSize; k++)
                for(int i = 0; i < m_dilations.size(); i++)
                {
                    int idx = k * m_dilations.size() + i;
                    std::tie( y, skip_y ) = m_blockStack[idx].forward( y );
                    skip_ys.block(idx * m_numChannels, 0, m_numChannels, x.cols()) = skip_y;
                }
            RowMatrixXf transpose_skip_ys = skip_ys.transpose();
            return m_outputLinear.forward( transpose_skip_ys ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override
        {
            auto conv_state_dict = state_dict[std::string("conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( conv_state_dict );
            for(int k = 0; k < m_stackSize; k++)
                for(int i = 0; i < m_dilations.size(); i++)
                {
                    int idx = k * m_dilations.size() + i;
                    auto block_state_dict = state_dict[std::string("blockStack.") + std::to_string(idx)].get<std::map<std::string, nlohmann::json>>();
                    m_blockStack[idx].loadStateDict( block_state_dict );
                }
            auto linear_state_dict = state_dict[std::string("linear")].get<std::map<std::string, nlohmann::json>>();
            m_outputLinear.loadStateDict( linear_state_dict );
        }

    private:
        int m_numChannels, m_stackSize;
        std::vector<int> m_dilations;
        CausalDilatedConv1d m_inputConv;
        std::vector<ResidualBlock> m_blockStack;
        Linear m_outputLinear;
    };
}