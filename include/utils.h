#pragma once

#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>
#include <assert.h>
#include <iostream>

namespace MicroTorch
{
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;

    inline float sigmoid( float xx ) { return 1.f / (1.f + std::exp(-xx) ); }

    inline float tanh( float xx ) { return std::tanh(xx); }

    inline std::vector<RowMatrixXf> loadTensor( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict[name].get<std::map<std::string, nlohmann::json>>();
        auto shape = data[std::string("shape")].get<std::vector<int>>();
        auto values = data[std::string("values")].get<std::vector<float>>();

        std::vector<RowMatrixXf> tensor;

        // Loop through the vector and assign elements to rows
        for (int i = 0; i < shape[0]; ++i)
        {
            // Create the Eigen matrix
            RowMatrixXf matrix(shape[1], shape[2]);
            for (int j = 0; j < shape[1]; ++j)
                for (int k = 0; k < shape[2]; ++k)
                    matrix(j, k) = values[i * shape[1] + j * shape[2] + k];
            tensor.push_back( matrix );
        }
        
        return tensor;
    }

    inline RowMatrixXf loadMatrix( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict[name].get<std::map<std::string, nlohmann::json>>();
        auto shape = data[std::string("shape")].get<std::vector<int>>();
        auto values = data[std::string("values")].get<std::vector<float>>();

        // Create the Eigen matrix
        RowMatrixXf matrix(shape[0], shape[1]);

        // Loop through the vector and assign elements to rows
        for (int i = 0; i < shape[0]; ++i)
            for (int j = 0; j < shape[1]; ++j)
                matrix(i, j) = values[i * shape[1] + j];

        return matrix;
    }

    inline Eigen::VectorXf loadVector( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict[name].get<std::map<std::string, nlohmann::json>>();
        auto shape = data[std::string("shape")].get<std::vector<int>>();
        auto values = data[std::string("values")].get<std::vector<float>>();
        return Eigen::Map<Eigen::VectorXf>( values.data(), shape[0] );
    } 

    inline Eigen::RowVectorXf convolve1d(const Eigen::RowVectorXf& in, const Eigen::RowVectorXf& weights)
    {
        assert(in.size() >= weights.size());
        int numCalculations = in.size() - weights.size() + 1;
        std::vector<float> values;
        for(int i = 0; i < numCalculations; i++)
            values.push_back( (in.segment(i, weights.size()).array() * weights.array()).sum() );
        return Eigen::Map<Eigen::VectorXf>( values.data(), numCalculations );
    }

}