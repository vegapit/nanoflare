#pragma once

#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "utils.h"

namespace MicroTorch
{

    class ConvClipper
    {
    public:
        ConvClipper(size_t kernel_size, size_t dilation) : m_inputConv( 1, 1, kernel_size, true, dilation ), m_outputConv( 1, 1, kernel_size, true, 1 ) {}
        ~ConvClipper() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            RowMatrixXf y = m_inputConv.forward( x );
            y.noalias() = y.cwiseMin( m_ceiling ).cwiseMax( m_floor );
            return m_outputConv.forward( y );
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto input_state_dict = state_dict[std::string("input_conv")].get<std::map<std::string, nlohmann::json>>();
            m_inputConv.loadStateDict( input_state_dict );
            auto output_state_dict = state_dict[std::string("output_conv")].get<std::map<std::string, nlohmann::json>>();
            m_outputConv.loadStateDict( output_state_dict );
            auto floor = loadVector( std::string("floor"), state_dict );
            setFloor( floor );
            auto ceiling = loadVector( std::string("ceiling"), state_dict );
            setCeiling( ceiling );
        }

    private:

        void setFloor(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == 1);
            m_floor = -1.f / (1.f + std::exp(-5.f * v(0)));
        }

        void setCeiling(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == 1);
            m_ceiling = 1.f / (1.f + std::exp(-5.f * v(0)));
        }

        CausalDilatedConv1d m_inputConv, m_outputConv;
        float m_floor, m_ceiling;
    };
}