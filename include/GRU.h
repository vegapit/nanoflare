#pragma once

#include <eigen3/Eigen/Dense>
#include "GRUCell.h"

namespace MicroTorch
{

    class GRU
    {
    public:
        GRU(int input_size, int hidden_size, bool bias) : m_cell(input_size, hidden_size, bias), m_h(Eigen::VectorXf::Zero(3*hidden_size)) {}
        ~GRU() = default;

        void reset()
        {
            m_h.setZero();
        }

        RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x )
        {
            assert(x.cols() == m_cell.getInputSize());
            RowMatrixXf y( x.rows(), m_cell.getHiddenSize() );
            for(int i = 0; i < x.rows(); i++)
            {
                RowMatrixXf row = x.row(i);
                m_cell.forward( row, m_h );
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
            
            if(m_cell.getBias())
            {
                auto bih = loadVector( std::string("bias_ih_l0"), state_dict );
                auto bhh = loadVector( std::string("bias_hh_l0"), state_dict );
                m_cell.setBiasIH( bih );
                m_cell.setBiasHH( bhh );
            }
        }
        
    private:
        Eigen::RowVectorXf m_h;
        GRUCell m_cell;
    };

}