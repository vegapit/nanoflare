#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "nanoflare/layers/Linear.h"
#include "nanoflare/layers/GRU.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/layers/Conv1d.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/utils.h"

using namespace Nanoflare;

constexpr int num_samples = 128;

// ---------------------------------------------------------------------------
// Linear / Dense
// ---------------------------------------------------------------------------

TEST_CASE("Linear 1->24")
{
    Linear nf(1, 24, true);
    RowMatrixXf x = RowMatrixXf::Random(1, num_samples);
    RowMatrixXf y = RowMatrixXf::Zero(24, num_samples);
    BENCHMARK("Nanoflare") { nf.forwardTranspose(x, y); return y(0, 0); };
}

TEST_CASE("Linear 24->24")
{
    Linear nf(24, 24, true);
    RowMatrixXf x = RowMatrixXf::Random(24, num_samples);
    RowMatrixXf y = RowMatrixXf::Zero(24, num_samples);
    BENCHMARK("Nanoflare") { nf.forwardTranspose(x, y); return y(0, 0); };
}

// ---------------------------------------------------------------------------
// GRU
// ---------------------------------------------------------------------------

TEST_CASE("GRU 1->64")
{
    GRU nf(1, 64, true);
    RowMatrixXf x = RowMatrixXf::Random(num_samples, 1);
    RowMatrixXf y = RowMatrixXf::Zero(num_samples, 64);
    BENCHMARK("Nanoflare") { nf.resetState(); nf.forward(x, y); return y(0, 0); };
}

// ---------------------------------------------------------------------------
// LSTM
// ---------------------------------------------------------------------------

TEST_CASE("LSTM 1->64")
{
    LSTM nf(1, 64, true);
    RowMatrixXf x = RowMatrixXf::Random(num_samples, 1);
    RowMatrixXf y = RowMatrixXf::Zero(num_samples, 64);
    BENCHMARK("Nanoflare") { nf.resetState(); nf.forward(x, y); return y(0, 0); };
}

// ---------------------------------------------------------------------------
// Conv1d (1x1 convolutions used in WaveNet post-processing)
// ---------------------------------------------------------------------------

TEST_CASE("Conv1d 8->8 k=1")
{
    Conv1d nf(8, 8, 1, true);
    RowMatrixXf x = RowMatrixXf::Random(8, num_samples);
    RowMatrixXf y = RowMatrixXf::Zero(8, num_samples);
    BENCHMARK("Nanoflare") { nf.forward(x, y); return y(0, 0); };
}

// ---------------------------------------------------------------------------
// CausalDilatedConv1d (WaveNet residual blocks)
// ---------------------------------------------------------------------------

TEST_CASE("CausalDilatedConv1d 8->8 k=3 d=1")
{
    CausalDilatedConv1d nf(8, 8, 3, true, 1);
    RowMatrixXf x = RowMatrixXf::Random(8, num_samples);
    RowMatrixXf y = RowMatrixXf::Zero(8, num_samples);
    BENCHMARK("Nanoflare") { nf.forward(x, y); return y(0, 0); };
}

TEST_CASE("CausalDilatedConv1d 8->8 k=3 d=8")
{
    CausalDilatedConv1d nf(8, 8, 3, true, 8);
    RowMatrixXf x = RowMatrixXf::Random(8, num_samples);
    RowMatrixXf y = RowMatrixXf::Zero(8, num_samples);
    BENCHMARK("Nanoflare") { nf.forward(x, y); return y(0, 0); };
}
