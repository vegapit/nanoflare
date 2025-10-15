#pragma once

#include "nanoflare/layers/Linear.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{

    class FiLM
    {
    public:
        FiLM(size_t feature_dim, size_t control_dim): m_scale(control_dim, feature_dim, true),
            m_shift(control_dim, feature_dim, true) {}
        ~FiLM() = default;

        inline void forward(const Eigen::Ref<RowMatrixXf>& x, const Eigen::Ref<RowMatrixXf>& params, Eigen::Ref<RowMatrixXf> y ) noexcept
        {
            m_scale.forward( params, y );
            y = y.cwiseProduct( x );

            RowMatrixXf shift = RowMatrixXf::Zero( params.rows(), x.cols() );
            m_shift.forward( params, shift );
            y += shift;
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto scale_state_dict = state_dict[std::string("scale")].get<std::map<std::string, nlohmann::json>>();
            m_scale.loadStateDict( scale_state_dict );
            auto shift_state_dict = state_dict[std::string("shift")].get<std::map<std::string, nlohmann::json>>();
            m_shift.loadStateDict( shift_state_dict );
        }

    private:
        Linear m_scale, m_shift;
    };
}