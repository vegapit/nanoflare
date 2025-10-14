#pragma once

#include "nanoflare/layers/Conv1d.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/layers/BatchNorm1d.h"

namespace Nanoflare
{

    class TCNBlock
    {
    public:
        TCNBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t dilation, bool use_batchnorm) noexcept
            : m_inChannels(in_channels), m_outChannels(out_channels), m_useBatchNorm(use_batchnorm),
            m_conv1( in_channels, out_channels, kernel_size, true, dilation ),
            m_conv2( out_channels, out_channels, kernel_size, true, 1 ),
            m_bn1( out_channels ),
            m_bn2( out_channels ),
            m_conv( in_channels, out_channels, 1, true )
        {}
        ~TCNBlock() = default;

        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y ) noexcept
        {
            if(x.data() == y.data())
            {
                RowMatrixXf temp( m_outChannels, x.cols());
                process( x, temp );
                y = std::move( temp );
            }
            else
            {
                if (y.rows() != m_outChannels || y.cols() != x.cols())
                    y.resize( m_outChannels, x.cols());
                process( x, y );
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
        }

        size_t getInChannels() { return m_inChannels; }
        size_t getOutChannels() { return m_outChannels; }

    private:

        inline void process( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> mat ) noexcept
        {
            m_conv1.forward( x, mat );
            if(m_useBatchNorm)
                m_bn1.apply( mat );
            Functional::LeakyReLU( mat, 0.2f );
            m_conv2.forward( mat, mat );
            if(m_useBatchNorm)
                m_bn2.apply( mat );
            Functional::LeakyReLU( mat, 0.2f );
            if(m_inChannels == m_outChannels)
                mat += x;
            else
            {
                if (m_temp.rows() != m_outChannels || m_temp.cols() != x.cols())
                    m_temp.resize(m_outChannels, x.cols());
                m_conv.forward( x, m_temp );
                mat += m_temp;
            }
        }

        bool m_useBatchNorm;
        CausalDilatedConv1d m_conv1, m_conv2;
        BatchNorm1d m_bn1, m_bn2;
        Conv1d m_conv;
        size_t m_inChannels, m_outChannels;
        RowMatrixXf m_temp;
    };
}