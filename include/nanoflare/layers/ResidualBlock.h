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
            m_inputConv(num_channels,
                gated ? 2 * num_channels : num_channels,
                kernel_size, true, dilation), 
            m_residualConv(num_channels, num_channels, 1, true),
            m_skipConv(num_channels, num_channels, 1, true)
        {}
        ~ResidualBlock() = default;

        inline std::pair<RowMatrixXf,RowMatrixXf> forward( const Eigen::Ref<const RowMatrixXf>& x ) const noexcept
        {
            if (m_z.rows() != m_numChannels || m_z.cols() != x.cols())
                m_z.resize(m_numChannels, x.cols());

            if (m_y_inner.rows() != (m_gated ? 2*m_numChannels : m_numChannels) || m_y_inner.cols() != x.cols())
                m_y_inner.resize(m_gated ? 2*m_numChannels : m_numChannels, x.cols());
            
            // Dilated causal conv
            m_y_inner = m_inputConv.forward( x );

            if(m_gated)
            {
                auto y_f = m_y_inner.topRows(m_numChannels);
                auto y_g = m_y_inner.bottomRows(m_numChannels);
                m_z.array() = y_f.array().tanh() * y_g.array().logistic();
            }
            else
                m_z.array() = m_y_inner.array().tanh();

            // project to residual and skip
            RowMatrixXf residual = m_residualConv.forward(m_z) + x;
            RowMatrixXf skip = m_skipConv.forward(m_z);
            return std::make_pair(residual, skip);
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto input_state_dict = state_dict[std::string("input_conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( input_state_dict );
            auto residual_state_dict = state_dict[std::string("residual_conv")].get<std::map<std::string, nlohmann::json>>();
            m_residualConv.loadStateDict(residual_state_dict);
            auto skip_state_dict = state_dict[std::string("skip_conv")].get<std::map<std::string, nlohmann::json>>();
            m_skipConv.loadStateDict(skip_state_dict);
        }

    private:
        CausalDilatedConv1d m_inputConv;
        Conv1d m_residualConv, m_skipConv;
        bool m_gated;
        size_t m_numChannels, m_kernelSize;
        mutable RowMatrixXf m_z, m_y_inner;
    };
}