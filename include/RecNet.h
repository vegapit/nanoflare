#pragma once

#include <eigen3/Eigen/Dense>
#include "LSTM.h"
#include "Linear.h"
#include <iostream>

namespace MicroTorch
{
    class RecNet
    {
    public:
        RecNet(int hidden_size) : m_hiddenSize(hidden_size), m_lstm(1, hidden_size, false), m_linear(hidden_size, 1, true) {}
        ~RecNet() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            RowMatrixXf transposed_x = x.transpose();
            RowMatrixXf y = m_lstm.forward( transposed_x );
            return x + m_linear.forward( y ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto lstm_state_dict = state_dict[std::string("lstm")].get<std::map<std::string, nlohmann::json>>();
            m_lstm.loadStateDict( lstm_state_dict );
            auto linear_state_dict = state_dict[std::string("linear")].get<std::map<std::string, nlohmann::json>>();
            m_linear.loadStateDict( linear_state_dict );
        }

    private:
        int m_hiddenSize;
        LSTM m_lstm;
        Linear m_linear;
    };

}