#pragma once

#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "utils.h"

namespace MicroTorch
{

    class ResidualBlock
    {
    public:
        ResidualBlock(int num_channels, int kernel_size, int dilation, bool input_bias, bool output_bias, bool gated, Activation activation) 
            : m_numChannels(num_channels), m_kernelSize(kernel_size), m_gated(gated), m_activation(activation),
            m_inputConv(num_channels, gated ? 2 * num_channels : num_channels, kernel_size, input_bias, dilation), 
            m_outputConv(num_channels, num_channels, 1, output_bias) {}
        ~ResidualBlock() = default;

        inline std::pair<RowMatrixXf,RowMatrixXf> forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            RowMatrixXf y_inner = m_inputConv.forward( x );
            
            RowMatrixXf y(m_numChannels, x.cols());
            
            if(m_gated)
            {
                RowMatrixXf y_filter = y_inner(Eigen::seqN(0, m_numChannels), Eigen::all);
                RowMatrixXf y_gate = y_inner(Eigen::seqN(m_numChannels, m_numChannels), Eigen::all);

                switch (m_activation)
                {
                case SIGMOID:
                    y.array() = y_filter.array().logistic();
                    break;
                case TANH:
                    y.array() = y_filter.array().tanh();
                    break;
                case SOFTSIGN:
                    y.array() = y_filter.array() / ( 1.f + y_filter.array().abs() );
                    break;
                }
                y.array() *= y_gate.array().logistic();
            }
            else
            {
                switch (m_activation)
                {
               case SIGMOID:
                    y.array() = y_inner.array().logistic();
                    break;
                case TANH:
                    y.array() = y_inner.array().tanh();
                    break;
                case SOFTSIGN:
                    y.array() = y_inner.array() / ( 1.f + y_inner.array().abs() );
                    break;
                }
            }

            y = m_outputConv.forward( y );

            return std::make_pair(y + x, y); // (Res,Skip)
        }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto input_state_dict = state_dict[std::string("inputConv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( input_state_dict );
            auto output_state_dict = state_dict[std::string("outputConv")].get<std::map<std::string, nlohmann::json>>();
            m_outputConv.loadStateDict( output_state_dict );
        }

    private:
        CausalDilatedConv1d m_inputConv;
        Conv1d m_outputConv;
        bool m_gated;
        int m_numChannels, m_kernelSize;
        Activation m_activation;
    };
}