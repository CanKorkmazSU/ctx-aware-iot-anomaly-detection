#pragma once
#include "genann.h"
#include <array>
#include <vector>
#include <cmath>
#include <limits>

// --- Configuration ---
constexpr int INPUT_DIM = 25;
constexpr int HIDDEN_DIM = 12;
constexpr int OUTPUT_DIM = 25;
constexpr float LR = 0.01f;
constexpr float ALA_LR = 0.05f; // Learning rate for the Trust Scores (Alphas)

class NanoFL {
private:
    genann* ann;

    std::array<double, INPUT_DIM> input_buf{};

    // Personalized Trust Scores (Alphas)
    // alpha = 1.0 (Trust Local), alpha = 0.0 (Trust Global)
    std::vector<float> alpha_hidden; // Size: HIDDEN_DIM
    std::vector<float> alpha_output; // Size: OUTPUT_DIM

public:
    NanoFL() {
        // Init Genann: inputs, 1 hidden layer, hidden size, outputs
        ann = genann_init(INPUT_DIM, 1, HIDDEN_DIM, OUTPUT_DIM);
        
        // Init Alphas to 0.5 (Neutral)
        alpha_hidden.assign(HIDDEN_DIM, 0.5f);
        alpha_output.assign(OUTPUT_DIM, 0.5f);
    }

    ~NanoFL() {
        if (ann) genann_free(ann);
    }

    // --- 1. Robust Training & Inference (Delegated to Genann) ---
    
    float predict(const std::vector<float>& input) {
        if (!ann || input.size() != static_cast<size_t>(INPUT_DIM)) {
            return std::numeric_limits<float>::infinity();
        }

        for (int i = 0; i < INPUT_DIM; i++) {
            input_buf[static_cast<size_t>(i)] = static_cast<double>(input[static_cast<size_t>(i)]);
        }

        // Returns Anomaly Score (MSE)
        // Genann returns a pointer to its internal output buffer
        const double* output = genann_run(ann, input_buf.data());
        
        double mse = 0.0;
        for (int i = 0; i < OUTPUT_DIM; i++) {
            double diff = static_cast<double>(input[static_cast<size_t>(i)]) - output[static_cast<size_t>(i)];
            mse += diff * diff;
        }
        return static_cast<float>(mse / static_cast<double>(OUTPUT_DIM));
    }

    void train(const std::vector<float>& data) {
        if (!ann || data.size() != static_cast<size_t>(INPUT_DIM)) return;

        for (int i = 0; i < INPUT_DIM; i++) {
            input_buf[static_cast<size_t>(i)] = static_cast<double>(data[static_cast<size_t>(i)]);
        }

        // Autoencoder: input is both feature and label
        genann_train(ann, input_buf.data(), input_buf.data(), static_cast<double>(LR));
    }

    // --- 2. The Complex Part: ALA on Genann's Flat Buffer ---
    
    // This function maps the Channel-Wise Alphas to the correct
    // chunks of Genann's flat weight array.
    void adaptive_aggregate(const std::vector<float>& global_weights, const std::vector<float>& val_sample) {
        if (!ann) return;
        if (global_weights.size() != static_cast<size_t>(ann->total_weights)) return; // Safety check
        if (val_sample.size() != static_cast<size_t>(OUTPUT_DIM)) return;

        // --- Step A: Update Alphas (Trust Scores) ---
        // (Simplified Heuristic: If we reconstruct better than Global, increase Alpha)
        // Note: For a 2-day deadline, we calculate this simply based on the previous predict()
        // For per-channel alpha updates, strictly we need to run global_weights inference too.
        // We will skip the global-inference for speed here and use a simple rule:
        // "If local error is low (< 0.05), trust local (increase alpha)."
        
        // We assume predict(val_sample) was just called, populating ann->output
        const double* local_out = ann->output;
        
        for(int i=0; i<OUTPUT_DIM; i++) {
            float err = std::abs(static_cast<float>(local_out[static_cast<size_t>(i)]) - val_sample[static_cast<size_t>(i)]);
            if(err < 0.05f) alpha_output[i] += ALA_LR;
            else alpha_output[i] -= ALA_LR;
            
            // Clamp [0.1, 0.9] - Never fully ignore anyone
            if(alpha_output[i] > 0.9f) alpha_output[i] = 0.9f;
            if(alpha_output[i] < 0.1f) alpha_output[i] = 0.1f;
        }

        // --- Step B: Apply Aggregation with Pointer Arithmetic ---
        // Genann Memory Layout:
        // [ Hidden Neuron 0 (Inputs+1 wts) ] [ Hidden Neuron 1 (Inputs+1 wts) ] ... 
        // [ Output Neuron 0 (Hidden+1 wts) ] [ Output Neuron 1 (Hidden+1 wts) ] ...
        
        double* local_w = ann->weight;
        const float* glob_w = global_weights.data();
        int w_idx = 0;

        // 1. Process Hidden Layer Weights
        // Stride = Inputs + 1 (Bias)
        int hidden_stride = INPUT_DIM + 1;
        
        for (int h = 0; h < HIDDEN_DIM; h++) {
            // Use fixed alpha=0.5 for hidden layer (Stability for bottleneck)
            float a = 0.5f; 
            
            for (int w = 0; w < hidden_stride; w++) {
                local_w[w_idx] = static_cast<double>(a) * local_w[w_idx]
                    + static_cast<double>(1.0f - a) * static_cast<double>(glob_w[w_idx]);
                w_idx++;
            }
        }

        // 2. Process Output Layer Weights
        // Stride = Hidden + 1 (Bias)
        int output_stride = HIDDEN_DIM + 1;

        for (int o = 0; o < OUTPUT_DIM; o++) {
            // Use Dynamic Alpha specific to this Output Channel (Pixel)
            float a = alpha_output[o]; 
            
            for (int w = 0; w < output_stride; w++) {
                local_w[w_idx] = static_cast<double>(a) * local_w[w_idx]
                    + static_cast<double>(1.0f - a) * static_cast<double>(glob_w[w_idx]);
                w_idx++;
            }
        }
    }

    // --- Helper to Export Weights ---
    std::vector<float> export_weights() {
        if (!ann) return {};
        std::vector<float> out;
        out.reserve(static_cast<size_t>(ann->total_weights));
        for (int i = 0; i < ann->total_weights; i++) {
            out.push_back(static_cast<float>(ann->weight[static_cast<size_t>(i)]));
        }
        return out;
    }
    
    // Helper to Import (Initialize Global Model buffer)
    void set_weights(const std::vector<float>& w) {
        if (!ann) return;
        if (w.size() != static_cast<size_t>(ann->total_weights)) return;

        for (int i = 0; i < ann->total_weights; i++) {
            ann->weight[static_cast<size_t>(i)] = static_cast<double>(w[static_cast<size_t>(i)]);
        }
    }
};