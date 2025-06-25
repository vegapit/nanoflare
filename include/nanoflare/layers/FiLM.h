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

        inline RowMatrixXf forward(const Eigen::Ref<RowMatrixXf>& x, const Eigen::Ref<RowMatrixXf>& params ) noexcept
        {
            return m_scale.forward( params ).cwiseProduct( x ) + m_shift.forward( params );
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