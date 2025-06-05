#pragma once

#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/MicroTCNBlock.h"
#include "nanoflare/layers/PlainSequential.h"
#include "nanoflare/utils.h"

namespace NanoFlare
{

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
        
        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            RowMatrixXf norm_x( x );
            normalise( norm_x );
            for(auto& block: m_blockStack )
                norm_x = block.forward( norm_x );
            norm_x.transposeInPlace();
            return m_plainSequential.forward( norm_x ).transpose();
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

    private:
        size_t m_hiddenSize, m_stackSize;
        std::vector<MicroTCNBlock> m_blockStack;
        PlainSequential m_plainSequential;
    };
}