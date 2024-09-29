#pragma once

#include <eigen3/Eigen/Dense>
#include "models/BaseModel.h"
#include "ConvClipper.h"
#include "utils.h"

namespace MicroTorch
{

    class SCC : public BaseModel
    {
    public:
        SCC(size_t kernel_size, size_t depth_size, float norm_mean, float norm_std) : BaseModel(norm_mean, norm_std), 
            m_kernelSize(kernel_size), m_depthSize(depth_size)
        {
            for(auto k = 0; k < depth_size; k++)
                m_stack.push_back( ConvClipper(kernel_size, std::pow(2, k)) );
        }
        ~SCC() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            RowMatrixXf norm_x( x );
            normalise( norm_x );
            for(auto& unit: m_stack )
                norm_x.noalias() = unit.forward( norm_x );
            return norm_x;
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            for(auto k = 0; k < m_depthSize; k++)
            {
                auto unit_state_dict = state_dict[std::string("stack.") + std::to_string(k)].get<std::map<std::string, nlohmann::json>>();
                m_stack[k].loadStateDict( unit_state_dict );
            }
        }

    private:
        size_t m_kernelSize, m_depthSize;
        std::vector<ConvClipper> m_stack;
    };
}