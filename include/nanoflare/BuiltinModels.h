#pragma once

#include "nanoflare/ModelBuilder.h"
#include "nanoflare/models/MicroTCN.h"
#include "nanoflare/models/ResRNN.h"
#include "nanoflare/models/TCN.h"
#include "nanoflare/models/WaveNet.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/layers/GRU.h"

namespace Nanoflare
{
    // Register builtin models during static initialization
    // This happens before main(), so models are available when needed
    namespace {
        inline bool registerBuiltinModels()
        {
            registerModel<MicroTCN>("MicroTCN");
            registerModel<ResRNN<GRU>>("ResGRU");
            registerModel<ResRNN<LSTM>>("ResLSTM");
            registerModel<TCN>("TCN");
            registerModel<WaveNet>("WaveNet");
            return true;
        }

        static const bool _builtinModelsRegistered = registerBuiltinModels();
    }
}
