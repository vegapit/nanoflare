#pragma once

#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "utils.h"

namespace MicroTorch
{

    class ResidualBlock
    {
    public:
        ResidualBlock(size_t num_channels, size_t kernel_size, size_t dilation, bool gated) 
            : m_numChannels(num_channels), m_kernelSize(kernel_size), m_gated(gated),
            m_inputConv(num_channels, gated ? 2 * num_channels : num_channels, kernel_size, true, dilation), 
            m_outputConv(num_channels, num_channels, 1, true) {}
        ~ResidualBlock() = default;

        inline std::pair<RowMatrixXf,RowMatrixXf> forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            RowMatrixXf y_inner = m_inputConv.forward( x );
            RowMatrixXf y(m_numChannels, x.cols());
            
            if(m_gated)
            {
                RowMatrixXf y_filter = y_inner(Eigen::seqN(0, m_numChannels), Eigen::all);
                RowMatrixXf y_gate = y_inner(Eigen::seqN(m_numChannels, m_numChannels), Eigen::all);
                y.array() = y_filter.array().tanh(); 
                y.array() *= y_gate.array().logistic();
            }
            else
                y.array() = y_inner.array().tanh();

            y.noalias() = m_outputConv.forward( y );

            return std::make_pair(y + x, y); // (Res,Skip)
        }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto input_state_dict = state_dict[std::string("input_conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( input_state_dict );
            auto output_state_dict = state_dict[std::string("output_conv")].get<std::map<std::string, nlohmann::json>>();
            m_outputConv.loadStateDict( output_state_dict );
        }

    private:
        CausalDilatedConv1d m_inputConv;
        Conv1d m_outputConv;
        bool m_gated;
        size_t m_numChannels, m_kernelSize;
    };
}