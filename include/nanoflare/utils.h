#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace Nanoflare
{

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;

    inline std::vector<RowMatrixXf> loadTensor( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<size_t>>();
        auto values = data.at("values").get<std::vector<float>>();

        std::vector<RowMatrixXf> tensor;

        // Loop through the vector and assign elements to rows
        for (auto i = 0; i < shape[0]; ++i)
        {
            // Create the Eigen matrix
            RowMatrixXf matrix(shape[1], shape[2]);
            for (auto j = 0; j < shape[1]; ++j)
                for (auto k = 0; k < shape[2]; ++k)
                    matrix(j, k) = values[i * shape[1] * shape[2] + j * shape[2] + k];
            tensor.push_back( matrix );
        }

        return tensor;
    }

    inline RowMatrixXf loadMatrix( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<size_t>>();
        auto values = data.at("values").get<std::vector<float>>();
        return Eigen::Map<RowMatrixXf>( values.data(), shape[0], shape[1] );
    }

    inline Eigen::VectorXf loadVector( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<size_t>>();
        auto values = data.at("values").get<std::vector<float>>();
        return Eigen::Map<Eigen::VectorXf>( values.data(), shape[0] );
    } 

    inline void convolve1d(const Eigen::Ref<const Eigen::RowVectorXf>& in, const Eigen::Ref<const Eigen::RowVectorXf>& weights, Eigen::Ref<Eigen::RowVectorXf> out) noexcept
    {
        size_t weights_size = weights.size();
        size_t out_size = in.size() - weights_size + 1;
        assert(out.size() == out_size);

        for (size_t i = 0; i < out_size; ++i)
            out(i) = in.segment(i, weights_size).cwiseProduct(weights).sum();
    }

    inline void dilatedcausalconvolve1d(const Eigen::Ref<const Eigen::RowVectorXf>& in, const Eigen::Ref<const Eigen::RowVectorXf>& weights, size_t dilation, Eigen::Ref<Eigen::RowVectorXf> out) noexcept
    {   
        size_t weights_size = weights.size();
        size_t left_padding = dilation * ( weights_size - 1 );
        size_t out_size = in.size();
        assert(out.size() == out_size);

        for(auto i = 0; i < out_size; i++)
        {
            out(i) = 0.f;
            for(auto k = 0; k < weights_size; k++)
                if(i + k * dilation >= left_padding) // Avoid adding zero inputs
                    out(i) += weights(k) * in(i - left_padding + k * dilation);
        }
    }

}