#pragma once

#include <eigen3/Eigen/Dense>
#include "BaseModel.h"
#include "Linear.h"
#include "utils.h"
#include <iostream>

namespace MicroTorch
{

    template<class T> // T can be of type LSTM or GRU
    class ResRNN : public BaseModel
    {
    public:
        ResRNN(size_t input_size, size_t hidden_size, size_t output_size, bool rnn_bias, bool linear_bias, float norm_mean, float norm_std) : BaseModel(norm_mean, norm_std), 
            m_rnn(input_size, hidden_size, rnn_bias), 
            m_linear(hidden_size, output_size, linear_bias) {}
        ~ResRNN() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override
        {
            RowMatrixXf transposed_x( x.transpose() );
            normalise( transposed_x );
            RowMatrixXf y = m_rnn.forward( transposed_x );
            return x + m_linear.forward(y).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override
        {
            auto lstm_state_dict = state_dict[std::string("rnn")].get<std::map<std::string, nlohmann::json>>();
            m_rnn.loadStateDict( lstm_state_dict );
            auto linear_state_dict = state_dict[std::string("linear")].get<std::map<std::string, nlohmann::json>>();
            m_linear.loadStateDict( linear_state_dict );
        }

        void resetState() { m_rnn.resetState(); }

    private:
        T m_rnn;
        Linear m_linear;
    };

}