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
    
    // Tanh compression for better dynamic range visibility
    // 100.0 is the "soft knee" speed - values below this are linear-ish, above are compressed
    float scale_factor = 100.0; 
    float normalized = tanh(val_centered / scale_factor);
    
    // Map [-1, 1] from tanh to [0, 1] for the fragment shader
    fragValue = normalized * 0.5 + 0.5;
}
