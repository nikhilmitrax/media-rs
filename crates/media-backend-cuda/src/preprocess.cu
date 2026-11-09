// ABI is mirrored by launch_preprocess in kernel.rs. All offsets/pitches are bytes.
__device__ float read_code(const unsigned char* data, unsigned long long pitch,
                          int x, int y, int component, int components, int p010) {
    const unsigned char* row = data + (unsigned long long)y * pitch;
    int index = x * components + component;
    return p010 ? (float)(((const unsigned short*)row)[index] >> 6) : (float)row[index];
}

__device__ float sample_plane(const unsigned char* data, unsigned long long pitch,
                             float x, float y, int width, int height,
                             int component, int components, int p010, int filter) {
    x = fminf(fmaxf(x, 0.0f), (float)(width - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(height - 1));
    if (filter != 1) {
        // Filter 0 preserves existing nearest/ties-up semantics. Filter 2 is
        // explicit UV-only nearest/ties-earlier; source coordinates are already
        // clamped, so ceil(-0.5) safely selects zero at the first sample.
        int nx = filter == 2 ? (int)ceilf(x - 0.5f) : (int)floorf(x + 0.5f);
        int ny = filter == 2 ? (int)ceilf(y - 0.5f) : (int)floorf(y + 0.5f);
        nx = nx < width ? nx : width - 1;
        ny = ny < height ? ny : height - 1;
        return read_code(data, pitch, nx, ny, component, components, p010);
    }
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1 < width ? x0 + 1 : x0;
    int y1 = y0 + 1 < height ? y0 + 1 : y0;
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = read_code(data, pitch, x0, y0, component, components, p010);
    float b = read_code(data, pitch, x1, y0, component, components, p010);
    float c = read_code(data, pitch, x0, y1, component, components, p010);
    float d = read_code(data, pitch, x1, y1, component, components, p010);
    return (a + fx * (b - a)) + fy * ((c + fx * (d - c)) - (a + fx * (b - a)));
}

__device__ void convert_rgb(float Y, float U, float V, float y_bias, float y_scale,
    float c_bias, float c_scale, float kr, float kb, float rgb[3]) {
    Y = (Y - y_bias) / y_scale; U = (U - c_bias) / c_scale; V = (V - c_bias) / c_scale;
    float kg = 1.0f - kr - kb;
    rgb[0] = fminf(fmaxf(Y + 2.0f * (1.0f - kr) * V, 0.0f), 1.0f);
    rgb[1] = fminf(fmaxf(Y - 2.0f * kb * (1.0f - kb) / kg * U - 2.0f * kr * (1.0f - kr) / kg * V, 0.0f), 1.0f);
    rgb[2] = fminf(fmaxf(Y + 2.0f * (1.0f - kb) * U, 0.0f), 1.0f);
}

__device__ void source_rgb(const unsigned char* luma, const unsigned char* chroma,
    unsigned long long y_pitch, unsigned long long uv_pitch, int x, int y,
    int width, int height, int p010, int chroma_filter, float chroma_x, float chroma_y,
    float y_bias, float y_scale, float c_bias, float c_scale, float kr, float kb, float rgb[3]) {
    float Y = read_code(luma, y_pitch, x, y, 0, 1, p010);
    float cx = ((float)x - chroma_x) * 0.5f, cy = ((float)y - chroma_y) * 0.5f;
    int cw = (width + 1) / 2, ch = (height + 1) / 2;
    float U = sample_plane(chroma, uv_pitch, cx, cy, cw, ch, 0, 2, p010, chroma_filter);
    float V = sample_plane(chroma, uv_pitch, cx, cy, cw, ch, 1, 2, p010, chroma_filter);
    convert_rgb(Y, U, V, y_bias, y_scale, c_bias, c_scale, kr, kb, rgb);
}

extern "C" __global__ void preprocess_yuv(
    const unsigned char* input, float* output, unsigned long long output_offset, unsigned long long uv_address, int input_width, int input_height,
    unsigned long long y_offset, unsigned long long y_pitch,
    unsigned long long uv_offset, unsigned long long uv_pitch,
    int crop_x, int crop_y, int crop_width, int crop_height,
    int width, int height, int p010, int nhwc, int bilinear, int chroma_filter, int resize_rgb,
    float y_bias, float y_scale, float c_bias, float c_scale, float kr, float kb,
    float chroma_x, float chroma_y,
    float mean_r, float mean_g, float mean_b, float std_r, float std_g, float std_b) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixels = (unsigned int)width * (unsigned int)height;
    if (index >= pixels) return;
    output += output_offset;
    int x = index % (unsigned int)width, y = index / (unsigned int)width;
    float sx = (float)crop_x + ((float)x + 0.5f) * (float)crop_width / (float)width - 0.5f;
    float sy = (float)crop_y + ((float)y + 0.5f) * (float)crop_height / (float)height - 0.5f;
    sx = fminf(fmaxf(sx, (float)crop_x), (float)(crop_x + crop_width - 1));
    sy = fminf(fmaxf(sy, (float)crop_y), (float)(crop_y + crop_height - 1));
    const unsigned char* luma = input + y_offset;
    const unsigned char* uv_plane = (const unsigned char*)uv_address + uv_offset;
    float rgb[3];
    if (resize_rgb) {
        if (bilinear) {
            int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
            int x1 = x0 + 1 < crop_x + crop_width ? x0 + 1 : x0;
            int y1 = y0 + 1 < crop_y + crop_height ? y0 + 1 : y0;
            float fx = sx - (float)x0, fy = sy - (float)y0;
            float a[3], b[3], c[3], d[3];
            source_rgb(luma, uv_plane, y_pitch, uv_pitch, x0, y0, input_width, input_height, p010, chroma_filter, chroma_x, chroma_y, y_bias, y_scale, c_bias, c_scale, kr, kb, a);
            source_rgb(luma, uv_plane, y_pitch, uv_pitch, x1, y0, input_width, input_height, p010, chroma_filter, chroma_x, chroma_y, y_bias, y_scale, c_bias, c_scale, kr, kb, b);
            source_rgb(luma, uv_plane, y_pitch, uv_pitch, x0, y1, input_width, input_height, p010, chroma_filter, chroma_x, chroma_y, y_bias, y_scale, c_bias, c_scale, kr, kb, c);
            source_rgb(luma, uv_plane, y_pitch, uv_pitch, x1, y1, input_width, input_height, p010, chroma_filter, chroma_x, chroma_y, y_bias, y_scale, c_bias, c_scale, kr, kb, d);
            for (int channel = 0; channel < 3; ++channel) {
                float top = a[channel] + fx * (b[channel] - a[channel]);
                float bottom = c[channel] + fx * (d[channel] - c[channel]);
                rgb[channel] = top + fy * (bottom - top);
            }
        } else {
            int nx = (int)floorf(sx + 0.5f), ny = (int)floorf(sy + 0.5f);
            nx = nx < crop_x + crop_width ? nx : crop_x + crop_width - 1;
            ny = ny < crop_y + crop_height ? ny : crop_y + crop_height - 1;
            source_rgb(luma, uv_plane, y_pitch, uv_pitch, nx, ny, input_width, input_height, p010, chroma_filter, chroma_x, chroma_y, y_bias, y_scale, c_bias, c_scale, kr, kb, rgb);
        }
    } else {
        float Y = sample_plane(luma, y_pitch, sx, sy, input_width, input_height, 0, 1, p010, bilinear);
        float cx = (sx - chroma_x) * 0.5f, cy = (sy - chroma_y) * 0.5f;
        int cw = (input_width + 1) / 2, ch = (input_height + 1) / 2;
        float U = sample_plane(uv_plane, uv_pitch, cx, cy, cw, ch, 0, 2, p010, chroma_filter);
        float V = sample_plane(uv_plane, uv_pitch, cx, cy, cw, ch, 1, 2, p010, chroma_filter);
        convert_rgb(Y, U, V, y_bias, y_scale, c_bias, c_scale, kr, kb, rgb);
    }
    float r = rgb[0], g = rgb[1], b = rgb[2];
    r = (fminf(fmaxf(r, 0.0f), 1.0f) - mean_r) / std_r;
    g = (fminf(fmaxf(g, 0.0f), 1.0f) - mean_g) / std_g;
    b = (fminf(fmaxf(b, 0.0f), 1.0f) - mean_b) / std_b;
    if (nhwc) {
        output[3ULL * index] = r; output[3ULL * index + 1] = g; output[3ULL * index + 2] = b;
    } else {
        output[index] = r; output[(unsigned long long)pixels + index] = g; output[2ULL * pixels + index] = b;
    }
}
