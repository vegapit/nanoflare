#pragma once

#include <eigen3/Eigen/Dense>
#include "GRUCell.h"

namespace MicroTorch
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
            Eigen::RowVectorXf row( x.cols() );
            for(Eigen::Index i = 0; i < x.rows(); i++)
            {
                row = x.row(i);
                m_cell.forward( row, m_h );
                y.row(i) = m_h; // Assign h to output
            }
            return y;
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict)
        {
            auto wih = loadMatrix( std::string("weight_ih"), state_dict );
            auto whh = loadMatrix( std::string("weight_hh"), state_dict );
            m_cell.setWeightIH( wih );
            m_cell.setWeightHH( whh );
            
            if(m_cell.isBiased())
            {
                auto bih = loadVector( std::string("bias_ih"), state_dict );
                auto bhh = loadVector( std::string("bias_hh"), state_dict );
                m_cell.setBiasIH( bih );
                m_cell.setBiasHH( bhh );
            }
        }
        
    private:
        Eigen::VectorXf m_h;
        GRUCell m_cell;
    };

}