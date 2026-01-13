#version 450

layout(location = 0) in float fragValue;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    layout(offset = 96) float trailWeight;
    layout(offset = 100) float contrast;
} push;

// Red-White-Green colormap with Intensity Scaling
// t is in [0, 1], where 0.5 is the neutral point (zero)
vec3 cold_hot(float t) {
    // Calculate distance from center [0, 1]
    float dist = abs(t - 0.5) * 2.0; 
    
    // Apply contrast correction
    // If contrast < 1.0 (e.g. 0.2), it boosts low values (log-like behavior)
    // If contrast > 1.0, it suppresses low values
    float intensity = pow(dist, push.contrast); 

    vec3 white = vec3(0.8, 0.8, 0.8);
    
    if (t < 0.5) {
        // Sellers (Negative) -> RED
        vec3 red = vec3(1.0, 0.1, 0.0);
        return mix(white, red, intensity);
    } else {
        // Buyers (Positive) -> GREEN
        vec3 green = vec3(0.2, 1.0, 0.0);
        return mix(white, green, intensity);
    }
}

void main() {
    // Input is already normalized to [0, 1] in vertex shader
    // 0.5 corresponds to value 0
    float t = clamp(fragValue, 0.0, 1.0);
    
    vec3 color = cold_hot(t);
    // Translucent points (use trailWeight/w from push constant)
    outColor = vec4(color, push.trailWeight);
}
