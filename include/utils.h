#pragma once

#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>
#include "xsimd/xsimd.hpp"

namespace MicroTorch
{
    enum Activation
    {
        SIGMOID,
        TANH,
        SOFTSIGN
    };

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;
    
    inline void xSigmoid( const Eigen::Ref<Eigen::VectorXf>& in, Eigen::Ref<Eigen::VectorXf> out )
    {
        using b_type = xsimd::batch<float>;
        std::size_t inc = b_type::size;
        std::size_t size = out.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc)
        {
            b_type xvec = b_type::load_unaligned(in.data() + i);
            b_type rvec = 1.f / (1.f + xsimd::exp(-xvec));
            rvec.store_unaligned(out.data() + i);
        }
        // Remaining part that cannot be vectorize
        for (std::size_t i = vec_size; i < size; ++i)
            out(i) = 1.f / (1.f + std::exp(-in(i)));
    }

    inline void xTanh( const Eigen::Ref<Eigen::VectorXf>& in, Eigen::Ref<Eigen::VectorXf> out )
    {
        using b_type = xsimd::batch<float>;
        std::size_t inc = b_type::size;
        std::size_t size = out.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc)
        {
            b_type xvec = b_type::load_unaligned(in.data() + i);
            b_type rvec = xsimd::tanh(xvec);
            rvec.store_unaligned(out.data() + i);
        }
        // Remaining part that cannot be vectorize
        for (std::size_t i = vec_size; i < size; ++i)
            out(i) = std::tanh(in(i));
    }

    inline void xSoftSign( const Eigen::Ref<Eigen::VectorXf>& in, Eigen::Ref<Eigen::VectorXf> out )
    {
        using b_type = xsimd::batch<float>;
        std::size_t inc = b_type::size;
        std::size_t size = out.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc)
        {
            b_type xvec = b_type::load_unaligned(in.data() + i);
            b_type rvec = xvec / (1.f + xsimd::abs(xvec));
            rvec.store_unaligned(out.data() + i);
        }
        // Remaining part that cannot be vectorize
        for (std::size_t i = vec_size; i < size; ++i)
            out(i) = in(i) / (1.f + std::abs(in(i)));
    }

    inline std::vector<RowMatrixXf> loadTensor( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<int>>();
        auto values = data.at("values").get<std::vector<float>>();

        std::vector<RowMatrixXf> tensor;

        // Loop through the vector and assign elements to rows
        for (int i = 0; i < shape[0]; ++i)
        {
            // Create the Eigen matrix
            RowMatrixXf matrix(shape[1], shape[2]);
            for (int j = 0; j < shape[1]; ++j)
                for (int k = 0; k < shape[2]; ++k)
                    matrix(j, k) = values[i * shape[1] * shape[2] + j * shape[2] + k];
            tensor.push_back( matrix );
        }

        return tensor;
    }

    inline RowMatrixXf loadMatrix( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<int>>();
        auto values = data.at("values").get<std::vector<float>>();
        return Eigen::Map<RowMatrixXf>(values.data(), shape[0], shape[1]);
    }

    inline Eigen::VectorXf loadVector( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<int>>();
        auto values = data.at("values").get<std::vector<float>>();
        return Eigen::Map<Eigen::VectorXf>( values.data(), shape[0] );
    } 

inline Eigen::RowVectorXf pad(const Eigen::RowVectorXf& in, int padding)
    {
        std::vector<float> values(in.size() + 2 * padding);
        for(int i = 0; i < values.size(); i++)
        {
            if((i < padding)||(i >= in.size() + padding))
                values[i] = 0.f; 
            else
                values[i] = in(i - padding); 
        }
        return Eigen::Map<Eigen::RowVectorXf>( values.data(), values.size() );
    }

    inline Eigen::RowVectorXf dilate(const Eigen::RowVectorXf& in, int dilation)
    {
        int size = dilation * (in.size() - 1) + 1;
        std::vector<float> values(size);
        for(int i = 0; i < in.size() - 1; i++)
            for(int k = 0; k < dilation; k++)
            {
                if( k == 0 )
                    values[dilation * i + k] = in[i];
                else
                    values[dilation * i + k] = 0.f; 
            }
        values[size - 1] = in(in.size() - 1);
        return Eigen::Map<Eigen::RowVectorXf>( values.data(), values.size() );
    }

    inline Eigen::RowVectorXf convolve1d(const Eigen::RowVectorXf& in, const Eigen::RowVectorXf& weights)
    {   
        int numCalculations = in.size();
        std::vector<float> values(numCalculations);
        for(int i = 0; i < numCalculations; i++)
            values[i] = in.segment(i, weights.size()).dot(weights);
        return Eigen::Map<Eigen::RowVectorXf>( values.data(), numCalculations );
    }

    enum ModelType {
        RES_LSTM,
        RES_GRU
    };

    NLOHMANN_JSON_SERIALIZE_ENUM( ModelType, {
        {RES_LSTM, "ResLSTM"},
        {RES_GRU, "ResGRU"}
    })

    struct ModelDef
    {
        ModelType type;
        int input_size, hidden_size, output_size;
        bool rnn_bias, linear_bias;
        float norm_mean, norm_std;
    };

    inline void from_json(const nlohmann::json& j, ModelDef& obj) {
        obj.type = j.at("type").template get<ModelType>();
        j.at("input_size").get_to(obj.input_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
        j.at("rnn_bias").get_to(obj.rnn_bias);
        j.at("linear_bias").get_to(obj.linear_bias);
        j.at("norm_mean").get_to(obj.norm_mean);
        j.at("norm_std").get_to(obj.norm_std);
    }

}