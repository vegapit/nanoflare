#pragma once

#include "Linear.h"

namespace MicroTorch
{

    class PlainSequential
    {
    public:
        PlainSequential(size_t in_channels, size_t out_channels, size_t hidden_channels, size_t num_hidden_layers) 
            : m_inChannels(in_channels), m_outChannels(out_channels),
            m_directLinear(in_channels, out_channels, true),
            m_inputLinear(in_channels, hidden_channels, true),
            m_outputLinear(hidden_channels, out_channels, true)
        {
            for(int i = 0; i < num_hidden_layers; i++)
                m_hiddenLinear.push_back( Linear(hidden_channels, hidden_channels, true) );
        }
        ~PlainSequential() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) const noexcept
        {
            RowMatrixXf y = m_inputLinear.forward( x ).cwiseMax(0.f);
            for(auto& linear: m_hiddenLinear)
                y = linear.forward( y ).cwiseMax(0.f);
            if(m_inChannels == m_outChannels)
                return x + m_outputLinear.forward( y );
            else
                return m_directLinear.forward( x ) + m_outputLinear.forward( y );
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto direct_linear_state_dict = state_dict[std::string("direct_linear")].get<std::map<std::string, nlohmann::json>>();
            m_directLinear.loadStateDict( direct_linear_state_dict );
            auto input_linear_state_dict = state_dict[std::string("input_linear")].get<std::map<std::string, nlohmann::json>>();
            m_inputLinear.loadStateDict( input_linear_state_dict );
            auto output_linear_state_dict = state_dict[std::string("output_linear")].get<std::map<std::string, nlohmann::json>>();
            m_outputLinear.loadStateDict( output_linear_state_dict );
            for(int i = 0; i < m_hiddenLinear.size(); i++)
            {
                auto hidden_linear_state_dict = state_dict[std::string("hidden_linear.") + std::to_string(i)].get<std::map<std::string, nlohmann::json>>();
                m_hiddenLinear[i].loadStateDict( hidden_linear_state_dict );
            }
        }

    private:
        Linear m_directLinear, m_inputLinear, m_outputLinear;
        std::vector<Linear> m_hiddenLinear;
        size_t m_inChannels, m_outChannels; 
    };
}