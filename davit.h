#ifndef DAVIT_H
#define DAVIT_H

#include "ggml.h"
#include <vector>

struct DaViTConfig {
    int image_size;
    int patch_size;
    int hidden_size;
    int n_intermediate;
    int projection_dim;
    int n_head;
    int n_layer;
    float eps;
    int n_groups; // for channel group attention
};

struct DaViTLayer {
    // Spatial Window Attention
    struct ggml_tensor * spatial_k_w;
    struct ggml_tensor * spatial_k_b;
    struct ggml_tensor * spatial_q_w;
    struct ggml_tensor * spatial_q_b;
    struct ggml_tensor * spatial_v_w;
    struct ggml_tensor * spatial_v_b;
    struct ggml_tensor * spatial_o_w;
    struct ggml_tensor * spatial_o_b;

    // Channel Group Attention
    struct ggml_tensor * channel_k_w;
    struct ggml_tensor * channel_k_b;
    struct ggml_tensor * channel_q_w;
    struct ggml_tensor * channel_q_b;
    struct ggml_tensor * channel_v_w;
    struct ggml_tensor * channel_v_b;
    struct ggml_tensor * channel_o_w;
    struct ggml_tensor * channel_o_b;

    // LayerNorms and FFNs
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
    struct ggml_tensor * ffn_1_w;
    struct ggml_tensor * ffn_1_b;
    struct ggml_tensor * ffn_2_w;
    struct ggml_tensor * ffn_2_b;
};

struct DaViTModel {
    DaViTConfig config;

    struct ggml_tensor * patch_embed;
    struct ggml_tensor * pos_embed;

    std::vector<DaViTLayer> layers;

    struct ggml_tensor * ln_post_w;
    struct ggml_tensor * ln_post_b;

    struct ggml_context * ctx;
};

DaViTModel * davit_model_load(const char * fname);
void davit_model_free(DaViTModel * model);

struct ggml_tensor * davit_forward(
    DaViTModel * model,
    struct ggml_context * ctx,
    struct ggml_tensor * input
);

#endif // DAVIT_H
