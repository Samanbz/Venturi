#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    // The 0.3 alpha combined with Additive Blending makes overlaps glow white
    outColor = vec4(fragColor, 0.3); 
}