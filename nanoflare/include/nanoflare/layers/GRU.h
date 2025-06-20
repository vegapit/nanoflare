#pragma once

#include <Eigen/Dense>
#include "nanoflare/layers/GRUCell.h"

namespace Nanoflare
{

    class GRU
    {
    public:
        GRU(size_t input_size, size_t hidden_size, bool bias) : m_cell(input_size, hidden_size, bias), m_h(Eigen::VectorXf::Zero(hidden_size)) {}
        ~GRU() = default;

        void resetState() { m_h.setZero(); }

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept
        {
            RowMatrixXf y( x.rows(), m_cell.getHiddenSize() );
            for(auto i = 0; i < x.rows(); i++)
            {
                m_cell.forward( x.row(i), m_h );
                y.row(i) = m_h; // Assign h to output
            }
            return y;
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto wih = loadMatrix( std::string("weight_ih_l0"), state_dict );
            auto whh = loadMatrix( std::string("weight_hh_l0"), state_dict );
            m_cell.setWeightIH( wih );
            m_cell.setWeightHH( whh );
            
            if(m_cell.isBiased())
            {
                auto bih = loadVector( std::string("bias_ih_l0"), state_dict );
                auto bhh = loadVector( std::string("bias_hh_l0"), state_dict );
                m_cell.setBiasIH( bih );
                m_cell.setBiasHH( bhh );
            }
        }
        
    private:
        Eigen::VectorXf m_h;
        GRUCell m_cell;
    };

}