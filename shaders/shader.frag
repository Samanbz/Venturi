#version 450

layout(location = 0) in float fragValue;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    layout(offset = 72) float trailWeight;
} push;

// Blue-White-Red colormap
// t is in [0, 1], where 0.5 is the neutral point (zero)
vec3 cold_hot(float t) {
    // Use slightly nicer colors than pure primaries
    vec3 blue = vec3(0.05, 0.05, 0.8);
    vec3 white = vec3(0.95, 0.95, 0.95);
    vec3 red = vec3(0.8, 0.05, 0.05);
    
    if (t < 0.5) {
        // Interpolate from Blue (0.0) to White (0.5)
        // Use smoothstep for nicer transition
        float local_t = t * 2.0;
        return mix(blue, white, local_t);
    } else {
        // Interpolate from White (0.5) to Red (1.0)
        float local_t = (t - 0.5) * 2.0;
        return mix(white, red, local_t);
    }
}

void main() {
    // Input is already normalized to [0, 1] in vertex shader
    // 0.5 corresponds to value 0
    float t = clamp(fragValue, 0.0, 1.0);
    
    vec3 color = cold_hot(t);
    // Apply trail weight to RGB, keep Alpha as is (or scale it too?)
    // If we scale RGB, it becomes darker.
    outColor = vec4(color * push.trailWeight, 0.3);
}
