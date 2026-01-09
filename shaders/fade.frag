#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    layout(offset = 80) vec2 scale;
    vec2 offset;
    float fadeRate;
} push;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 readUV = inUV * push.scale + push.offset;
    
    // Bounds check for the "read" coordinate
    if (readUV.x < 0.0 || readUV.x > 1.0 || readUV.y < 0.0 || readUV.y > 1.0) {
         outColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
         vec4 prev = texture(texSampler, readUV);
         outColor = vec4(prev.rgb * (1.0 - push.fadeRate), 1.0);
    }
}
