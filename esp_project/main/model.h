#pragma once
#include "genann.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// --- Configuration (10x10 Grid) ---
constexpr int INPUT_DIM = 100;
constexpr int HIDDEN_DIM = 48;
constexpr int OUTPUT_DIM = 100;
constexpr float LR = 0.01f;
constexpr float ALA_LR = 0.05f;

class NanoFL {
  genann *ann;
  std::vector<float>
      alphas; // Consolidated: [0..HIDDEN-1] = Hidden, [HIDDEN..END] = Output
  std::vector<float> mixed_buf;

public:
  NanoFL() {
    ann = genann_init(INPUT_DIM, 1, HIDDEN_DIM, OUTPUT_DIM);
    alphas.assign(HIDDEN_DIM + OUTPUT_DIM, 0.8f); // Init trust: 80% Local
    mixed_buf.resize(ann->total_weights);
  }

  ~NanoFL() {
    if (ann)
      genann_free(ann);
  }

  float predict(const std::vector<float> &in) {
    if (in.size() != INPUT_DIM)
      return 999.0f;
    std::array<double, INPUT_DIM> buf;
    for (int i = 0; i < INPUT_DIM; i++)
      buf[i] = in[i]; // float->double

    const double *out = genann_run(ann, buf.data());

    // MSE Calculation
    double mse = 0.0;
    for (int i = 0; i < OUTPUT_DIM; i++) {
      double diff = in[i] - out[i];
      mse += diff * diff;
    }
    return (float)(mse / OUTPUT_DIM);
  }

  void train(const std::vector<float> &in) {
    if (in.size() != INPUT_DIM)
      return;
    std::array<double, INPUT_DIM> buf;
    for (int i = 0; i < INPUT_DIM; i++)
      buf[i] = in[i];
    genann_train(ann, buf.data(), buf.data(),
                 LR); // Autoencoder: Target = Input
  }

  // --- FedALA: Neuron-Wise Adaptive Aggregation ---
  void adaptive_aggregate(const std::vector<float> &global_w,
                          const std::vector<std::vector<float>> &val_data) {
    if (global_w.size() != ann->total_weights || val_data.empty())
      return;

    std::vector<float> local_w = get_weights();

    // Optimize Alpha for each neuron (Hidden then Output)
    // Structure: [HIDDEN_LAYER (In->Hid)] followed by [OUTPUT_LAYER (Hid->Out)]
    int w_idx = 0;
    int a_idx = 0;

    // 1. Hidden Layer Neurons
    for (int h = 0; h < HIDDEN_DIM; h++, a_idx++) {
      optimize_alpha(a_idx, w_idx, INPUT_DIM + 1, local_w, global_w, val_data);
      w_idx += (INPUT_DIM + 1);
    }

    // 2. Output Layer Neurons
    for (int o = 0; o < OUTPUT_DIM; o++, a_idx++) {
      optimize_alpha(a_idx, w_idx, HIDDEN_DIM + 1, local_w, global_w, val_data);
      w_idx += (HIDDEN_DIM + 1);
    }

    // Final Apply
    mix_weights(local_w, global_w);
    set_weights(mixed_buf);
  }

  std::vector<float> get_weights() {
    std::vector<float> w(ann->total_weights);
    for (int i = 0; i < ann->total_weights; i++)
      w[i] = (float)ann->weight[i];
    return w;
  }

  void set_weights(const std::vector<float> &w) {
    if (w.size() == ann->total_weights)
      for (int i = 0; i < ann->total_weights; i++)
        ann->weight[i] = w[i];
  }

private:
  float evaluate_batch(const std::vector<std::vector<float>> &batch) {
    float loss = 0;
    for (const auto &sample : batch)
      loss += predict(sample);
    return loss / batch.size();
  }

  // Update specific block of weights using alpha
  void mix_block(int start_w, int count, float alpha,
                 const std::vector<float> &loc,
                 const std::vector<float> &glob) {
    for (int i = 0; i < count; i++) {
      mixed_buf[start_w + i] =
          alpha * loc[start_w + i] + (1.0f - alpha) * glob[start_w + i];
    }
  }

  // Re-generate entire mixed_buf based on current alphas
  void mix_weights(const std::vector<float> &loc,
                   const std::vector<float> &glob) {
    int w_idx = 0;
    int a_idx = 0;
    // Hidden
    for (int h = 0; h < HIDDEN_DIM; h++, a_idx++) {
      mix_block(w_idx, INPUT_DIM + 1, alphas[a_idx], loc, glob);
      w_idx += INPUT_DIM + 1;
    }
    // Output
    for (int o = 0; o < OUTPUT_DIM; o++, a_idx++) {
      mix_block(w_idx, HIDDEN_DIM + 1, alphas[a_idx], loc, glob);
      w_idx += HIDDEN_DIM + 1;
    }
  }

  void optimize_alpha(int alpha_idx, int w_idx, int stride,
                      const std::vector<float> &loc,
                      const std::vector<float> &glob,
                      const std::vector<std::vector<float>> &val) {
    float &a = alphas[alpha_idx];
    float a_orig = a;

    // Finite Difference Gradients
    float eps = 0.05f;

    // Try A + eps
    a = std::max(0.1f, std::min(0.9f, a_orig + eps));
    mix_weights(loc, glob); // Re-mix everything
    set_weights(mixed_buf);
    float l_plus = evaluate_batch(val);

    // Try A - eps
    a = std::max(0.1f, std::min(0.9f, a_orig - eps));
    mix_weights(loc, glob);
    set_weights(mixed_buf);
    float l_minus = evaluate_batch(val);

    // Update Alpha
    if (std::abs(a_orig + eps - (a_orig - eps)) > 1e-6) {
      float grad = (l_plus - l_minus) / (2 * eps);
      a_orig -= ALA_LR * grad;
      a = std::max(0.1f, std::min(0.9f, a_orig));
    } else {
      a = a_orig; // Restore if degenerate
    }
    alphas[alpha_idx] = a;
  }
};