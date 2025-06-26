#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/ConvClipper.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    struct ConvWaveshaperParameters
    {
        size_t kernel_size, depth_size, num_channels;
    };

    inline void from_json(const nlohmann::json& j, ConvWaveshaperParameters& obj) {
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("depth_size").get_to(obj.depth_size);
        j.at("num_channels").get_to(obj.num_channels);
    }

    class ConvWaveshaper : public BaseModel
    {
    public:
        ConvWaveshaper(size_t kernel_size, size_t depth_size, size_t num_channels, float norm_mean, float norm_std) : BaseModel(norm_mean, norm_std), 
            m_kernelSize(kernel_size), m_depthSize(depth_size)
        {
            m_stack.push_back( ConvClipper(1, num_channels, kernel_size, 1) );
            for(auto k = 1; k < depth_size - 1; k++)
                m_stack.push_back( ConvClipper(num_channels, num_channels, kernel_size, std::pow(2, k)) );
            m_stack.push_back( ConvClipper(num_channels, 1, kernel_size, std::pow(2, depth_size - 1)) );    
        }
        ~ConvWaveshaper() = default;
        
        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            RowMatrixXf norm_x( x );
            normalise( norm_x );
            for(auto& unit: m_stack )
                norm_x = unit.forward( norm_x );
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

        static void build(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();
            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();
            auto parameters = data.at("parameters").template get<ConvWaveshaperParameters>();
            model = std::make_shared<ConvWaveshaper>(parameters.kernel_size, parameters.depth_size, parameters.num_channels, config.norm_mean, config.norm_std);
            model->loadStateDict( state_dict );
        }

    private:
        size_t m_kernelSize, m_depthSize;
        std::vector<ConvClipper> m_stack;
    };

}