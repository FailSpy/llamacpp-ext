#include "davit.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>

DaViTModel * davit_model_load(const char * fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open file: ") + fname);
    }

    DaViTModel * model = new DaViTModel();
    file.read(reinterpret_cast<char*>(&model->config), sizeof(DaViTConfig));

    // Initialize ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    model->ctx = ggml_init(params);

    // Load patch embedding
    model->patch_embed = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32,
                                             model->config.hidden_size,
                                             (model->config.image_size / model->config.patch_size) * (model->config.image_size / model->config.patch_size) * 3);
    file.read(reinterpret_cast<char*>(model->patch_embed->data), ggml_nbytes(model->patch_embed));

    // Load position embedding
    model->pos_embed = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32,
                                           model->config.hidden_size,
                                           (model->config.image_size / model->config.patch_size) * (model->config.image_size / model->config.patch_size) + 1);
    file.read(reinterpret_cast<char*>(model->pos_embed->data), ggml_nbytes(model->pos_embed));

    // Load layers
    model->layers.resize(model->config.n_layer);
    for (int i = 0; i < model->config.n_layer; ++i) {
        auto & layer = model->layers[i];

        // Load spatial attention weights
        layer.spatial_q_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);
        layer.spatial_k_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);
        layer.spatial_v_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);
        layer.spatial_o_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);

        // Load channel attention weights
        layer.channel_q_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);
        layer.channel_k_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);
        layer.channel_v_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);
        layer.channel_o_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.hidden_size);

        // Load FFN weights
        layer.ffn_1_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.n_intermediate, model->config.hidden_size);
        layer.ffn_2_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, model->config.n_intermediate);

        // Load LayerNorm weights
        layer.ln_1_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, 1);
        layer.ln_2_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, 1);

        // Read all weights
        file.read(reinterpret_cast<char*>(layer.spatial_q_w->data), ggml_nbytes(layer.spatial_q_w));
        file.read(reinterpret_cast<char*>(layer.spatial_k_w->data), ggml_nbytes(layer.spatial_k_w));
        file.read(reinterpret_cast<char*>(layer.spatial_v_w->data), ggml_nbytes(layer.spatial_v_w));
        file.read(reinterpret_cast<char*>(layer.spatial_o_w->data), ggml_nbytes(layer.spatial_o_w));
        file.read(reinterpret_cast<char*>(layer.channel_q_w->data), ggml_nbytes(layer.channel_q_w));
        file.read(reinterpret_cast<char*>(layer.channel_k_w->data), ggml_nbytes(layer.channel_k_w));
        file.read(reinterpret_cast<char*>(layer.channel_v_w->data), ggml_nbytes(layer.channel_v_w));
        file.read(reinterpret_cast<char*>(layer.channel_o_w->data), ggml_nbytes(layer.channel_o_w));
        file.read(reinterpret_cast<char*>(layer.ffn_1_w->data), ggml_nbytes(layer.ffn_1_w));
        file.read(reinterpret_cast<char*>(layer.ffn_2_w->data), ggml_nbytes(layer.ffn_2_w));
        file.read(reinterpret_cast<char*>(layer.ln_1_w->data), ggml_nbytes(layer.ln_1_w));
        file.read(reinterpret_cast<char*>(layer.ln_2_w->data), ggml_nbytes(layer.ln_2_w));
    }

    // Load final layer norm weights
    model->ln_post_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->config.hidden_size, 1);
    file.read(reinterpret_cast<char*>(model->ln_post_w->data), ggml_nbytes(model->ln_post_w));

    file.close();
    return model;
}

void davit_model_free(DaViTModel * model) {
    if (model->ctx) {
        ggml_free(model->ctx);
    }
    delete model;
}

static struct ggml_tensor * spatial_window_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const DaViTLayer & layer,
    const DaViTConfig & config
) {
    // Reshape input: [N, C, H, W] -> [N, H, W, C]
    x = ggml_transpose(ctx, x);

    // Window partition
    int window_size = config.patch_size * 7;  // As per the paper
    int H = config.image_size / config.patch_size;
    int W = config.image_size / config.patch_size;
    int num_windows = (H / window_size) * (W / window_size);

    // Corrected ggml_view_4d call
    struct ggml_tensor * windows = ggml_view_4d(ctx, x,
                                                window_size, window_size, config.hidden_size, num_windows,
                                                window_size * ggml_element_size(x),
                                                window_size * W * ggml_element_size(x),
                                                H * W * ggml_element_size(x),
                                                0);

    // Compute Q, K, V
    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.spatial_q_w, windows);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.spatial_k_w, windows);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.spatial_v_w, windows);

    // Compute attention scores
    struct ggml_tensor * scores = ggml_mul_mat(ctx, ggml_transpose(ctx, k), q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrt(config.hidden_size / config.n_head));
    scores = ggml_soft_max(ctx, scores);

    // Apply attention
    struct ggml_tensor * attention = ggml_mul_mat(ctx, v, scores);

    // Merge windows
    attention = ggml_reshape_4d(ctx, attention,
                                config.hidden_size, window_size, window_size, num_windows * ggml_n_dims(x));
    attention = ggml_cont(ctx, ggml_transpose(ctx, attention));

    // Project output
    attention = ggml_mul_mat(ctx, layer.spatial_o_w, attention);

    return attention;
}

static struct ggml_tensor * channel_group_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const DaViTLayer & layer,
    const DaViTConfig & config
) {
    // Reshape and transpose: [N, C, H, W] -> [N, G, C/G, H*W]
    int H = config.image_size / config.patch_size;
    int W = config.image_size / config.patch_size;
    int G = config.n_groups;
    int C_per_G = config.hidden_size / G;

    struct ggml_tensor * x_reshaped = ggml_reshape_4d(ctx, x,
                                                      H * W, C_per_G, G, ggml_n_dims(x));
    x_reshaped = ggml_transpose(ctx, x_reshaped);

    // Compute Q, K, V
    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.channel_q_w, x_reshaped);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.channel_k_w, x_reshaped);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.channel_v_w, x_reshaped);

    // Compute attention scores
    struct ggml_tensor * scores = ggml_mul_mat(ctx, ggml_transpose(ctx, k), q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrt(C_per_G));
    scores = ggml_soft_max(ctx, scores);

    // Apply attention
    struct ggml_tensor * attention = ggml_mul_mat(ctx, v, scores);

    // Reshape back to original dimensions
    attention = ggml_transpose(ctx, attention);
    attention = ggml_reshape_4d(ctx, attention,
                                config.hidden_size, H, W, ggml_n_dims(x) > 3 ? x->ne[3] : 1);

    // Project output
    attention = ggml_mul_mat(ctx, layer.channel_o_w, attention);

    return attention;
}

struct ggml_tensor * davit_forward(
    DaViTModel * model,
    struct ggml_context * ctx,
    struct ggml_tensor * input
) {
    struct ggml_tensor * x = input;

    // Patch embedding
    x = ggml_mul_mat(ctx, model->patch_embed, x);

    // Add position embedding
    x = ggml_add(ctx, x, model->pos_embed);

    // Process layers
    for (const auto & layer : model->layers) {
        struct ggml_tensor * residual = x;

        // Layer Norm 1
        x = ggml_norm(ctx, x, model->config.eps);
        x = ggml_mul(ctx, x, layer.ln_1_w);

        // Spatial Window Attention
        struct ggml_tensor * spatial_out = spatial_window_attention(ctx, x, layer, model->config);
        x = ggml_add(ctx, x, spatial_out);

        // Channel Group Attention
        struct ggml_tensor * channel_out = channel_group_attention(ctx, x, layer, model->config);
        x = ggml_add(ctx, x, channel_out);

        // Add residual
        x = ggml_add(ctx, x, residual);

        residual = x;

        // Layer Norm 2
        x = ggml_norm(ctx, x, model->config.eps);
        x = ggml_mul(ctx, x, layer.ln_2_w);

        // FFN
        x = ggml_mul_mat(ctx, layer.ffn_1_w, x);
        x = ggml_gelu(ctx, x);
        x = ggml_mul_mat(ctx, layer.ffn_2_w, x);

        // Add residual
        x = ggml_add(ctx, x, residual);
    }

    // Final layer norm
    x = ggml_norm(ctx, x, model->config.eps);
    x = ggml_mul(ctx, x, model->ln_post_w);

    return x;
}
