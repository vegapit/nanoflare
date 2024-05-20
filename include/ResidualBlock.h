#pragma once

#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "utils.h"

namespace MicroTorch
{

    class ResidualBlock
    {
    public:
        ResidualBlock(int num_channels, int kernel_size, int dilation, bool input_bias, bool output_bias, Activation activation_filter, Activation activation_gate) 
            : m_numChannels(num_channels), m_kernelSize(kernel_size), m_activationFilter(activation_filter), m_activationGate(activation_gate),
            m_inputConv(num_channels, 2 * num_channels, kernel_size, input_bias, dilation), 
            m_outputConv(num_channels, num_channels, 1, output_bias) {}
        ~ResidualBlock() = default;

        inline std::pair<RowMatrixXf,RowMatrixXf> forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            RowMatrixXf y_inner = m_inputConv.forward( x );
            
            RowMatrixXf y_filter = y_inner(Eigen::seqN(0, m_numChannels), Eigen::all);
            RowMatrixXf y_gate = y_inner(Eigen::seqN(m_numChannels, m_numChannels), Eigen::all);

            RowMatrixXf y_f(y_filter.rows(), y_filter.cols());
            RowMatrixXf y_g(y_gate.rows(), y_gate.cols());

            switch(m_activationFilter)
            {
                case Activation::SIGMOID: xSigmoid( y_filter, y_f); break;
                case Activation::TANH: xTanh( y_filter, y_f); break;
                case Activation::SOFTSIGN: xSoftSign( y_filter, y_f); break;
            }

            switch(m_activationGate)
            {
                case Activation::SIGMOID: xSigmoid( y_gate, y_g); break;
                case Activation::TANH: xTanh( y_gate, y_g); break;
                case Activation::SOFTSIGN: xSoftSign( y_gate, y_g); break;
            }

            RowMatrixXf y = y_f.cwiseProduct( y_g );
            y = m_outputConv.forward( y );

            return std::make_pair(y + x, y);
        }
        
        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto input_state_dict = state_dict[std::string("inputConv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( input_state_dict );
            auto output_state_dict = state_dict[std::string("outputConv")].get<std::map<std::string, nlohmann::json>>();
            m_outputConv.loadStateDict( output_state_dict );
        }

    private:
        CausalDilatedConv1d m_inputConv;
        Conv1d m_outputConv;
        Activation m_activationFilter, m_activationGate;
        int m_numChannels, m_kernelSize;
    };
}