#pragma once

#include "nanoflare/layers/Conv1d.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/layers/BatchNorm1d.h"
#include "nanoflare/layers/PReLU.h"

namespace NanoFlare
{

    class MicroTCNBlock
    {
    public:
        MicroTCNBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t dilation) 
            : m_inChannels(in_channels), m_outChannels(out_channels),
            m_conv1( in_channels, out_channels, kernel_size, true, dilation ),
            m_bn1( out_channels ),
            m_f1( out_channels ),
            m_conv( in_channels, out_channels, 1, true )
        {}
        ~MicroTCNBlock() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            auto y = m_conv1.forward( x );
            m_f1.apply( y );
            m_bn1.apply( y );
            if(m_inChannels == m_outChannels)
                return x + y;
            else 
                return m_conv.forward( x ) + y;
        }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto conv_state_dict = state_dict[std::string("conv")].get<std::map<std::string, nlohmann::json>>();
            m_conv.loadStateDict( conv_state_dict );
            auto conv1_state_dict = state_dict[std::string("conv1")].get<std::map<std::string, nlohmann::json>>();
            m_conv1.loadStateDict( conv1_state_dict );
            auto bn1_state_dict = state_dict[std::string("bn1")].get<std::map<std::string, nlohmann::json>>();
            m_bn1.loadStateDict( bn1_state_dict );
            auto f1_state_dict = state_dict[std::string("f1")].get<std::map<std::string, nlohmann::json>>();
            m_f1.loadStateDict( f1_state_dict );
        }

    private:
        CausalDilatedConv1d m_conv1;
        BatchNorm1d m_bn1;
        PReLU m_f1;
        Conv1d m_conv;
        size_t m_inChannels, m_outChannels; 
    };
}