#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/ResidualBlock.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/layers/PlainSequential.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{

    class WaveNet : public BaseModel
    {
    public:
        WaveNet(size_t input_size, size_t num_channels, size_t output_size, size_t kernel_size, std::vector<size_t> dilations, size_t stack_size, bool gated, size_t ps_hidden_size, size_t ps_num_hidden_layers, float norm_mean, float norm_std) : 
            BaseModel(norm_mean, norm_std), m_numChannels(num_channels), m_dilations(dilations), m_stackSize(stack_size), m_gated(gated),
            m_inputConv(input_size, num_channels, kernel_size, true, 1),
            m_plainSequential(num_channels, output_size, ps_hidden_size, ps_num_hidden_layers)
        {
            for(size_t k = 0; k < stack_size; k++)
                for(auto dilation: dilations)
                    m_blockStack.push_back( ResidualBlock(num_channels, kernel_size, dilation, gated) );
        }
        ~WaveNet() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            auto dilations_size = m_dilations.size();

            RowMatrixXf norm_x( x );
            normalise( norm_x );
            RowMatrixXf y = m_inputConv.forward( norm_x );

            RowMatrixXf skip_sum = RowMatrixXf::Zero( m_numChannels, x.cols() );
            RowMatrixXf skip_y( m_numChannels, x.cols() );
            for(auto k = 0; k < m_stackSize; k++)
                for(auto i = 0; i < dilations_size; i++)
                {
                    std::tie( y, skip_y ) = m_blockStack[k * dilations_size + i].forward( y );
                    skip_sum += skip_y;
                }

            RowMatrixXf res = skip_sum.cwiseMax(0.f).transpose();
            return m_plainSequential.forward( res ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            auto conv_state_dict = state_dict[std::string("conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( conv_state_dict );
            for(size_t k = 0; k < m_stackSize; k++)
                for(size_t i = 0; i < m_dilations.size(); i++)
                {
                    size_t idx = k * m_dilations.size() + i;
                    auto block_state_dict = state_dict[std::string("block_stack.") + std::to_string(idx)].get<std::map<std::string, nlohmann::json>>();
                    m_blockStack[idx].loadStateDict( block_state_dict );
                }
            auto ps_state_dict = state_dict[std::string("plain_sequential")].get<std::map<std::string, nlohmann::json>>();
            m_plainSequential.loadStateDict( ps_state_dict );
        }

    private:
        size_t m_numChannels, m_stackSize;
        bool m_gated;
        std::vector<size_t> m_dilations;
        CausalDilatedConv1d m_inputConv;
        std::vector<ResidualBlock> m_blockStack;
        PlainSequential m_plainSequential;
    };

    struct WaveNetParameters
    {
        size_t input_size, num_channels, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers;
        bool gated;
        std::vector<size_t> dilations;
    };

    inline void from_json(const nlohmann::json& j, WaveNetParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("num_channels").get_to(obj.num_channels);
        j.at("output_size").get_to(obj.output_size);
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("dilations").get_to(obj.dilations);
        j.at("stack_size").get_to(obj.stack_size);
        j.at("gated").get_to(obj.gated);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }
}