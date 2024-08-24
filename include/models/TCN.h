#pragma once

#include <eigen3/Eigen/Dense>
#include "BaseModel.h"
#include "TCNBlock.h"
#include "Linear.h"
#include "utils.h"

namespace MicroTorch
{

    class TCN : public BaseModel
    {
    public:
        TCN(size_t input_size, size_t output_size, size_t kernel_size, size_t stack_size, float norm_mean, float norm_std) : 
            BaseModel(norm_mean, norm_std), m_stackSize(stack_size),
            m_linear(std::pow(2, stack_size), output_size, true)
        {
            auto full_size = std::pow( 2, stack_size );
            for(auto k = 0; k <= stack_size; k++)
                m_blockStack.push_back( TCNBlock((k == 0) ? 1 : full_size, full_size, kernel_size, std::pow(2, k)) );
        }
        ~TCN() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            RowMatrixXf norm_x( x );
            normalise( norm_x );
            for(auto& block: m_blockStack )
                norm_x.noalias() = block.forward( norm_x );
            norm_x.transposeInPlace();
            return m_linear.forward( norm_x ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            for(auto k = 0; k <= m_stackSize; k++)
            {
                auto block_state_dict = state_dict[std::string("blockStack.") + std::to_string(k)].get<std::map<std::string, nlohmann::json>>();
                m_blockStack[k].loadStateDict( block_state_dict );
            }
            auto linear_state_dict = state_dict[std::string("linear")].get<std::map<std::string, nlohmann::json>>();
            m_linear.loadStateDict( linear_state_dict );
        }

    private:
        size_t m_stackSize;
        std::vector<TCNBlock> m_blockStack;
        Linear m_linear;
    };

}