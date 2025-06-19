#pragma once

#include "nanoflare/layers/Conv1d.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/utils.h"

namespace Nanoflare
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
            auto y_inner = m_inputConv.forward( x );
            
            RowMatrixXf y(m_numChannels, x.cols());
            if(m_gated)
                y.array() = y_inner.topRows(m_numChannels).array().tanh() * y_inner.bottomRows(m_numChannels).array().logistic();
            else
                y.array() = y_inner.array().tanh();

            y = m_outputConv.forward( y );

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