#pragma once

#include <Eigen/Dense>
#include "nanoflare/utils.h"

namespace Nanoflare
{
    class PReLU
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PReLU(size_t num_channels) : m_numChannels(num_channels), m_w(Eigen::RowVectorXf::Zero(num_channels)) {}
        ~PReLU() = default;

        inline void apply( Eigen::Ref<RowMatrixXf> x ) const noexcept
        {
            for( auto col: x.colwise() )
                col = col.cwiseMax( 0.f ) + col.cwiseMin( 0.f ).cwiseProduct( m_w.transpose() );
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto w = loadVector( std::string("weight"), state_dict );
            setWeight( w );
        }

    private:
        
        void setWeight(const Eigen::Ref<Eigen::RowVectorXf>& v)
        {
            assert(v.size() == m_numChannels);
            m_w = v;
        }

        Eigen::RowVectorXf m_w;
        size_t m_numChannels;
    };
}