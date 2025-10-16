#pragma once

#include <cassert>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/Conv1d.h"
#include "nanoflare/layers/ResidualBlock.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    struct WaveNetParameters
    {
        size_t input_size, num_channels, output_size, kernel_size, stack_size, hidden_size;
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
        j.at("hidden_size").get_to(obj.hidden_size);
    }

    class WaveNet : public BaseModel
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        WaveNet(size_t input_size, size_t num_channels, size_t output_size, size_t kernel_size, std::vector<size_t> dilations, size_t stack_size, bool gated, size_t hidden_size, float norm_mean, float norm_std) : 
            BaseModel(norm_mean, norm_std, input_size, output_size), 
            m_numChannels(num_channels), m_dilations(dilations), m_stackSize(stack_size), m_gated(gated),
            m_inputConv(input_size, num_channels, kernel_size, true, 1),
            m_postConv1(num_channels, hidden_size, 1, true),
            m_postConv2(hidden_size, output_size, 1, true)
        {
            for(size_t k = 0; k < stack_size; k++)
                for(auto dilation: dilations)
                    m_blockStack.push_back( ResidualBlock(num_channels, kernel_size, dilation, gated) );
        }
        ~WaveNet() = default;

        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y ) noexcept override final
        {
            assert((y.rows() == m_postConv2.getOutChannels() && y.cols() == x.cols()) && "WaveNet.forward: Wrong output shape");

            auto dilations_size = m_dilations.size();
            auto skip_scale = 1.f / std::sqrt( static_cast<float>(m_stackSize * dilations_size) );

            m_norm_x = x;
            normalise( m_norm_x );

            // CausalDilatedConv: input(C_in, time) output(C_numCh, time)
            if (m_temp.rows() != m_numChannels || m_temp.cols() != x.cols())
                m_temp.resize(m_numChannels, x.cols());
            m_inputConv.forward( m_norm_x, m_temp );

            // ResidualBlock: input(C_numCh, time) output(C_numCh, time)
            if (m_skip_sum.rows() != m_numChannels|| m_skip_sum.cols() != x.cols())
                m_skip_sum.resize(m_numChannels, x.cols());
            if (m_skip_temp.rows() != m_numChannels || m_skip_temp.cols() != x.cols())
                m_skip_temp.resize(m_numChannels, x.cols());
            m_skip_sum.setZero();
            for(auto k = 0; k < m_stackSize; k++)
                for(auto i = 0; i < dilations_size; i++)
                {
                    m_blockStack[k * dilations_size + i].forward( m_temp, m_temp, m_skip_temp );
                    m_skip_sum += m_skip_temp;
                }
            m_skip_sum *= skip_scale;
            Functional::ReLU( m_skip_sum );
            
            if (m_temp_hidden.rows() != m_postConv1.getOutChannels() || m_temp_hidden.cols() != x.cols())
                m_temp_hidden.resize(m_postConv1.getOutChannels(), x.cols());
            m_postConv1.forward( m_skip_sum, m_temp_hidden );
            Functional::ReLU( m_temp_hidden );
            
            m_postConv2.forward( m_temp_hidden, y );
            denormalise( y );
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            auto input_conv_state_dict = state_dict[std::string("input_conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( input_conv_state_dict );
            auto post_conv1_state_dict = state_dict[std::string("post_conv1")].get<std::map<std::string, nlohmann::json>>();
            m_postConv1.loadStateDict( post_conv1_state_dict );
            auto post_conv2_state_dict = state_dict[std::string("post_conv2")].get<std::map<std::string, nlohmann::json>>();
            m_postConv2.loadStateDict( post_conv2_state_dict );

            for(size_t k = 0; k < m_stackSize; k++)
                for(size_t i = 0; i < m_dilations.size(); i++)
                {
                    size_t idx = k * m_dilations.size() + i;
                    auto block_state_dict = state_dict[std::string("block_stack.") + std::to_string(idx)].get<std::map<std::string, nlohmann::json>>();
                    m_blockStack[idx].loadStateDict( block_state_dict );
                }
        }

        static void build(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();

            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();
            auto parameters = data.at("parameters").template get<WaveNetParameters>();
            model = std::make_shared<WaveNet>(parameters.input_size, parameters.num_channels, parameters.output_size, parameters.kernel_size, parameters.dilations, parameters.stack_size, parameters.gated, parameters.hidden_size, config.norm_mean, config.norm_std);
            model->loadStateDict( state_dict ); 
        }

    private:
        size_t m_numChannels, m_stackSize;
        bool m_gated;
        std::vector<size_t> m_dilations;
        CausalDilatedConv1d m_inputConv;
        Conv1d m_postConv1, m_postConv2;
        std::vector<ResidualBlock> m_blockStack;
        mutable RowMatrixXf m_temp, m_skip_temp, m_skip_sum, m_norm_x, m_temp_hidden;
    };

}