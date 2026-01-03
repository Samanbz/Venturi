#version 450

layout(location = 0) in float inX;
layout(location = 1) in float inY;
layout(location = 2) in float inValue;

layout(location = 0) out float fragValue;

layout(push_constant) uniform PushConstants {
    mat4 projection;
    float minColor;
    float maxColor;
} push;

void main() {
    gl_Position = push.projection * vec4(inX, inY, 0.0, 1.0);
    gl_PointSize = 1.0;
    
    float center = (push.maxColor + push.minColor) * 0.5;
    float half_range = (push.maxColor - push.minColor) * 0.5;
    if (abs(half_range) < 0.00001) half_range = 1.0;

    float val_centered = inValue - center;
    
    // Apply non-linear compression (sqrt) to spread out values near zero
    // This helps distinguish intermediate values from the neutral zero
    // We clamp the input to the range first to avoid sqrt of large numbers dominating
    float normalized = clamp(val_centered / half_range, -1.0, 1.0);
    float norm_signed = sign(normalized) * sqrt(abs(normalized));
    
    // Map [-1, 1] to [0, 1]
    fragValue = norm_signed * 0.5 + 0.5;
}
