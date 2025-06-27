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

        inline std::pair<RowMatrixXf,RowMatrixXf> forward( const Eigen::Ref<const RowMatrixXf>& x ) const noexcept
        {
            if (m_y.rows() != m_numChannels || m_y.cols() != x.cols())
                m_y.resize(m_numChannels, x.cols());

            if (m_y_inner.rows() != m_numChannels || m_y_inner.cols() != x.cols())
                m_y_inner.resize(m_numChannels, x.cols());

            m_y_inner = m_inputConv.forward( x );
            if(m_gated)
                m_y.array() = m_y_inner.topRows(m_numChannels).array().tanh() * m_y_inner.bottomRows(m_numChannels).array().logistic();
            else
                m_y.array() = m_y_inner.array().tanh();

            m_y = m_outputConv.forward( m_y );

            return std::make_pair(m_y + x, m_y); // (Res,Skip)
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
        mutable RowMatrixXf m_y, m_y_inner;
    };
}