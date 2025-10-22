#pragma once

#include "nanoflare/layers/Linear.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{

    class FiLM
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FiLM(size_t feature_dim, size_t control_dim)
            : m_feature_dim(feature_dim),
            m_control_dim(control_dim),
            m_scale(control_dim, feature_dim, true),
            m_shift(control_dim, feature_dim, true),
            m_gamma(Eigen::RowVectorXf::Zero(feature_dim)),
            m_beta(Eigen::RowVectorXf::Zero(feature_dim))
        {}

        ~FiLM() = default;

        inline void forward(const Eigen::Ref<const RowMatrixXf>& x, 
                        const Eigen::Ref<const Eigen::RowVectorXf>& params, 
                        Eigen::Ref<RowMatrixXf> y) noexcept
        {
            // params: (1, control_dim) gamma: (1, feature_dim)
            m_scale.forward(params, m_gamma);
            m_shift.forward(params, m_beta);

            // x: (time, feature_dim) gamma: (1, feature_dim)
            y.noalias() = (x.array().rowwise() * m_gamma.array()).matrix();
            y.rowwise() += m_beta;
        }
        
        inline void forwardTranspose(const Eigen::Ref<const RowMatrixXf>& x, 
                                    const Eigen::Ref<const Eigen::RowVectorXf>& params, 
                                    Eigen::Ref<RowMatrixXf> y) noexcept
        {
            // params: (1, control_dim) gamma: (1, feature_dim)
            m_scale.forward(params, m_gamma);
            m_shift.forward(params, m_beta);
            
            const Eigen::VectorXf gamma_t = m_gamma.transpose();
            const Eigen::VectorXf beta_t = m_beta.transpose();
            
            // x: (feature_dim, time) gamma: (1, feature_dim)
            y.noalias() = (x.array().colwise() * gamma_t.array()).matrix();
            y.colwise() += beta_t;
        }

        void loadStateDict(const std::map<std::string, nlohmann::json>& state_dict)
        {
            auto scale_state_dict = state_dict.at("scale").get<std::map<std::string, nlohmann::json>>();
            m_scale.loadStateDict(scale_state_dict);
            
            auto shift_state_dict = state_dict.at("shift").get<std::map<std::string, nlohmann::json>>();
            m_shift.loadStateDict(shift_state_dict);
        }

        size_t getFeatureDim() const { return m_feature_dim; }
        size_t getControlDim() const { return m_control_dim; }

    private:
        size_t m_feature_dim, m_control_dim;
        Linear m_scale, m_shift;
        
        // Pre-allocated buffers (mutable for const-correctness in forward)
        mutable Eigen::RowVectorXf m_gamma, m_beta;
    };
}