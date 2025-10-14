#pragma once

#include "nanoflare/layers/Conv1d.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/layers/BatchNorm1d.h"
#include "nanoflare/layers/PReLU.h"

namespace Nanoflare
{

    class TCNBlock
    {
    public:
        TCNBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t dilation) 
            : m_inChannels(in_channels), m_outChannels(out_channels),
            m_conv1( in_channels, out_channels, kernel_size, true, dilation ),
            m_conv2( out_channels, out_channels, kernel_size, true, 1 ),
            m_bn1( out_channels ),
            m_bn2( out_channels ),
            m_f1( out_channels ),
            m_f2( out_channels ),
            m_conv( in_channels, out_channels, 1, true )
        {}
        ~TCNBlock() = default;

        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, RowMatrixXf& y ) noexcept
        {
            if(x.data() == y.data())
            {
                RowMatrixXf temp(m_outChannels, x.cols());
                m_conv1.forward( x, temp );
                m_bn1.apply( temp );
                m_f1.apply( temp );
                m_conv2.forward( temp, temp );
                m_bn2.apply( temp );
                m_f2.apply( temp );
                if(m_inChannels == m_outChannels)
                    temp += x;
                else
                {
                    if (m_temp.rows() != m_outChannels || m_temp.cols() != x.cols())
                        m_temp.resize(m_outChannels, x.cols());
                    m_conv.forward( x, m_temp );
                    temp += m_temp;
                }
                y = std::move(temp);
            }
            else
            {
                if (y.rows() != m_outChannels || y.cols() != x.cols())
                    y.resize( m_outChannels, x.cols());
                m_conv1.forward( x, y );
                m_bn1.apply( y );
                m_f1.apply( y );
                m_conv2.forward( y, y );
                m_bn2.apply( y );
                m_f2.apply( y );
                if(m_inChannels == m_outChannels)
                    y += x;
                else
                {
                    if (m_temp.rows() != m_outChannels || m_temp.cols() != x.cols())
                        m_temp.resize(m_outChannels, x.cols());
                    m_conv.forward( x, m_temp );
                    y += m_temp;
                }
            }
        }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto conv_state_dict = state_dict[std::string("conv")].get<std::map<std::string, nlohmann::json>>();
            m_conv.loadStateDict( conv_state_dict );
            auto conv1_state_dict = state_dict[std::string("conv1")].get<std::map<std::string, nlohmann::json>>();
            m_conv1.loadStateDict( conv1_state_dict );
            auto conv2_state_dict = state_dict[std::string("conv2")].get<std::map<std::string, nlohmann::json>>();
            m_conv2.loadStateDict( conv2_state_dict );
            auto bn1_state_dict = state_dict[std::string("bn1")].get<std::map<std::string, nlohmann::json>>();
            m_bn1.loadStateDict( bn1_state_dict );
            auto bn2_state_dict = state_dict[std::string("bn2")].get<std::map<std::string, nlohmann::json>>();
            m_bn2.loadStateDict( bn2_state_dict );
            auto f1_state_dict = state_dict[std::string("f1")].get<std::map<std::string, nlohmann::json>>();
            m_f1.loadStateDict( f1_state_dict );
            auto f2_state_dict = state_dict[std::string("f2")].get<std::map<std::string, nlohmann::json>>();
            m_f2.loadStateDict( f2_state_dict );
        }

    private:
        CausalDilatedConv1d m_conv1, m_conv2;
        BatchNorm1d m_bn1, m_bn2;
        PReLU m_f1, m_f2;
        Conv1d m_conv;
        size_t m_inChannels, m_outChannels;
        RowMatrixXf m_temp;
    };
}