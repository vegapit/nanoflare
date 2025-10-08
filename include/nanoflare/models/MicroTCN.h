#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/MicroTCNBlock.h"
#include "nanoflare/layers/PlainSequential.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    struct MicroTCNParameters
    {
        size_t input_size, hidden_size, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers;
    };

    inline void from_json(const nlohmann::json& j, MicroTCNParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("stack_size").get_to(obj.stack_size);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }

    class MicroTCN : public BaseModel
    {
    public:
        MicroTCN(size_t input_size, size_t hidden_size, size_t output_size, size_t kernel_size, size_t stack_size, size_t ps_hidden_size, size_t ps_num_hidden_layers, float norm_mean, float norm_std) : 
            BaseModel(norm_mean, norm_std), m_hiddenSize(hidden_size), m_stackSize(stack_size),
            m_plainSequential(hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers)
        {
            for(auto k = 0; k < stack_size; k++)
                m_blockStack.push_back( MicroTCNBlock((k == 0) ? input_size : hidden_size, hidden_size, kernel_size, std::pow(2, k)) );
        }
        ~MicroTCN() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<const RowMatrixXf>& x ) noexcept override final
        {
            m_norm_x = x;
            normalise( m_norm_x );
            for(auto& block: m_blockStack )
                m_norm_x = block.forward( m_norm_x );
            return m_plainSequential.forwardTranspose( m_norm_x );
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            for(auto k = 0; k < m_stackSize; k++)
            {
                auto block_state_dict = state_dict[std::string("block_stack.") + std::to_string(k)].get<std::map<std::string, nlohmann::json>>();
                m_blockStack[k].loadStateDict( block_state_dict );
            }
            auto ps_state_dict = state_dict[std::string("plain_sequential")].get<std::map<std::string, nlohmann::json>>();
            m_plainSequential.loadStateDict( ps_state_dict );
        }

        static void build(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();

            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();
            auto parameters = data.at("parameters").template get<MicroTCNParameters>();
            model = std::make_shared<MicroTCN>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.kernel_size, parameters.stack_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            model->loadStateDict( state_dict ); 
        }

    private:
        size_t m_hiddenSize, m_stackSize;
        std::vector<MicroTCNBlock> m_blockStack;
        PlainSequential m_plainSequential;
        mutable RowMatrixXf m_norm_x;
    };

}