#version 450

layout(location = 0) in float inX;
layout(location = 1) in float inY;

// --- FIX: Declare the output variable here ---
layout(location = 0) out vec3 fragColor; 

layout(push_constant) uniform PushConstants {
    mat4 projection;
} push;

void main() {
    // 1. Position
    gl_Position = push.projection * vec4(inX, inY, 0.0, 1.0);
    
    // 2. Make them bigger!
    gl_PointSize = 1.0; 

    // 3. Fake a "Heatmap" based on distance from center
    // Normalized approx distance (tweak divisor based on your bounds)
    // Using 20000.0 since your new boundaries are roughly [-50000, 50000]
    float dist = length(vec2(inX, inY)) / 20000.0; 
    
    // Low energy = Blue, High Energy = Red/Orange
    vec3 cold = vec3(0.1, 0.4, 1.0); // Electric Blue
    vec3 hot = vec3(1.0, 0.3, 0.1);  // Orange/Red
    
    // --- FIX: Now this assignment works ---
    fragColor = mix(cold, hot, clamp(dist, 0.0, 1.0));
}