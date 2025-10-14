#pragma once

#include "nanoflare/Functional.h"
#include "nanoflare/layers/Linear.h"

namespace Nanoflare
{

    class PlainSequential
    {
    public:
        PlainSequential(size_t in_channels, size_t out_channels, size_t hidden_channels, size_t num_hidden_layers) 
            : m_inChannels(in_channels), m_outChannels(out_channels), m_hiddenChannels(hidden_channels),
            m_directLinear(in_channels, out_channels, false),
            m_inputLinear(in_channels, hidden_channels, true),
            m_outputLinear(hidden_channels, out_channels, true)
        {
            for(int i = 0; i < num_hidden_layers; i++)
                m_hiddenLinear.push_back( Linear(hidden_channels, hidden_channels, true) );
        }
        ~PlainSequential() = default;

        inline void forward( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y ) noexcept
        {
            if (m_temp.rows() != x.rows() || m_temp.cols() != m_hiddenChannels)
                m_temp.resize(x.rows(), m_hiddenChannels);

            m_inputLinear.forward( x, m_temp );
            Functional::ReLU( m_temp );
            for(auto& linear: m_hiddenLinear)
            {
                linear.forward( m_temp, m_temp );
                Functional::ReLU( m_temp );
            }

            if(m_y.rows() != x.rows() || m_y.cols() != m_outChannels)
                m_y.resize(x.rows(), m_outChannels);
        
            m_outputLinear.forward( m_temp, m_y );

            if(m_inChannels == m_outChannels)
                m_y += x;
            else
            {
                if (m_temp.rows() != x.rows() || m_temp.cols() != m_outChannels)
                    m_temp.resize(x.rows(), m_outChannels);
                m_directLinear.forward( x, m_temp );
                m_y += m_temp;
            }

            y = m_y;
        }

        inline void forwardTranspose( const Eigen::Ref<const RowMatrixXf>& x, Eigen::Ref<RowMatrixXf> y ) noexcept
        {
            if (m_temp.rows() != m_hiddenChannels || m_temp.cols() != x.cols())
                m_temp.resize(m_hiddenChannels, x.cols());

            m_inputLinear.forwardTranspose( x, m_temp );
            Functional::ReLU( m_temp );
            for(auto& linear: m_hiddenLinear)
            {
                linear.forwardTranspose( m_temp, m_temp );
                Functional::ReLU( m_temp );
            }

            if(m_y.rows() != m_outChannels || m_y.cols() != x.cols())
                m_y.resize(m_outChannels, x.cols());

            m_outputLinear.forwardTranspose( m_temp, m_y );

            if(m_inChannels == m_outChannels)
                m_y += x;
            else
            {
                if (m_temp.rows() != m_outChannels || m_temp.cols() != x.cols())
                    m_temp.resize(m_outChannels, x.cols());
                m_directLinear.forwardTranspose( x, m_temp );
                m_y += m_temp;
            }

            y = m_y;
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

        size_t getInChannels() { return m_inChannels; }
        size_t getOutChannels() { return m_outChannels; }
        
    private:
        Linear m_directLinear, m_inputLinear, m_outputLinear;
        std::vector<Linear> m_hiddenLinear;
        size_t m_inChannels, m_outChannels, m_hiddenChannels;
        RowMatrixXf m_temp, m_y;
    };
}