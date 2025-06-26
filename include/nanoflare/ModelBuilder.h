#pragma once

#include <nlohmann/json.hpp>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/models/MicroTCN.h"
#include "nanoflare/models/ResRNN.h"
#include "nanoflare/models/ConvWaveshaper.h"
#include "nanoflare/models/TCN.h"
#include "nanoflare/models/WaveNet.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/layers/GRU.h"

namespace Nanoflare
{

    class ModelBuilder
    {
    public:
        // Define a type for the builder function
        using BuildFn = std::function<void(const nlohmann::json&, std::shared_ptr<BaseModel>&)>;

        // Get the singleton instance
        static ModelBuilder& getInstance() {
            static ModelBuilder instance; // Guaranteed to be destroyed correctly
            return instance;
        }

        // Register a builder function for a given shape name
        bool registerBuilder(const std::string& name, BuildFn builder) {
            // Return false if a builder for this name already exists
            return m_builders.insert({name, builder}).second;
        }

        // Create a mdoel by its string name
        void buildModel(const nlohmann::json& data, std::shared_ptr<BaseModel>& obj) {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();
            auto config = data.at("config").template get<ModelConfig>();
            auto it = m_builders.find( config.model_type );
            if (it != m_builders.end())
                it->second( data, obj ); // Call the registered builder function
        }

    private:
        ModelBuilder() = default; // Private constructor for singleton
        ~ModelBuilder() = default;
        ModelBuilder(const ModelBuilder&) = delete; // Delete copy constructor
        ModelBuilder& operator=(const ModelBuilder&) = delete; // Delete assignment operator

        std::map<std::string, BuildFn> m_builders;
    };

    template<typename T>
    inline void registerModel(const std::string& name) {
        ModelBuilder::getInstance().registerBuilder(name, &T::build);
    }

}