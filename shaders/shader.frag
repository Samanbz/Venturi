#version 450

layout(location = 0) in float fragValue;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    layout(offset = 72) float trailWeight;
} push;

// Red-White-Green colormap with Intensity Scaling
// t is in [0, 1], where 0.5 is the neutral point (zero)
vec3 cold_hot(float t) {
    // Calculate distance from center [0, 1]
    float dist = abs(t - 0.5) * 2.0; 
    
    // Non-linear visual scaling: boost low values so they aren't just white
    // power < 1.0 makes small values rise faster (more color)
    float intensity = pow(dist, 0.7); 

    vec3 white = vec3(0.95, 0.95, 0.95);
    
    if (t < 0.5) {
        // Sellers (Negative) -> RED
        vec3 red = vec3(1.0, 0.0, 0.0);
        return mix(white, red, intensity);
    } else {
        // Buyers (Positive) -> GREEN
        vec3 green = vec3(0.0, 1.0, 0.0);
        return mix(white, green, intensity);
    }
}

void main() {
    // Input is already normalized to [0, 1] in vertex shader
    // 0.5 corresponds to value 0
    float t = clamp(fragValue, 0.0, 1.0);
    
    vec3 color = cold_hot(t);
    outColor = vec4(color * push.trailWeight, 0.3);
}
