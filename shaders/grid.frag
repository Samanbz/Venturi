#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 p;
    float minC;
    float maxC;
    float w;
    float pad;
    vec2 worldSize;   // Width, Height
    vec2 worldOrigin; // MinX, MinY (Top-Left or Bottom-Left depending on coord sys)
    float trailWeight;
    float contrast;
} push;

void main() {
  
    // World Position calculation
    vec2 worldPos = push.worldOrigin + inUV * push.worldSize;
    
    // We want component-wise derivatives to handle anisotropic scaling (aspect ratio mismatch)
    vec2 derivative = vec2(fwidth(worldPos.x), fwidth(worldPos.y));

    // Auto-spacing based on derivatives independently
    float targetPixelSpacing = push.w;
    if (targetPixelSpacing <= 0.1) targetPixelSpacing = 100.0; // Default if not set

    vec2 targetWorldSpacing = targetPixelSpacing * derivative;
    
    // Find nearest power of 10 for each axis
    vec2 logStep = log(targetWorldSpacing) / log(10.0);
    vec2 baseStep = pow(vec2(10.0), floor(logStep));
    
    // Subdivisions
    vec2 mainStep = baseStep; 
    
    if (baseStep.x / targetWorldSpacing.x < 0.2) mainStep.x *= 5.0; 
    if (baseStep.x / targetWorldSpacing.x > 0.8) mainStep.x /= 2.0; 

    if (baseStep.y / targetWorldSpacing.y < 0.2) mainStep.y *= 5.0; 
    if (baseStep.y / targetWorldSpacing.y > 0.8) mainStep.y /= 2.0; 
    
    // Distance to nearest line (component-wise)
    // dist.x is dist to nearest vertical line
    // dist.y is dist to nearest horizontal line
    vec2 dist = abs(fract(worldPos / mainStep + 0.5) - 0.5) * mainStep;
    
    // Line width: 0.75 pixels * WorldPerPixel (separate for X and Y)
    vec2 lineWidth = 1.0 * derivative; 
    
    float gridAlpha = 0.0;
    
    if (dist.x < lineWidth.x || dist.y < lineWidth.y) {
        // Compute alpha for X (vertical lines) and Y (horizontal lines) separately
        float alphaX = (dist.x < lineWidth.x) ? (1.0 - smoothstep(0.0, lineWidth.x, dist.x)) : 0.0;
        float alphaY = (dist.y < lineWidth.y) ? (1.0 - smoothstep(0.0, lineWidth.y, dist.y)) : 0.0;
        
        // Combine (Max)
        gridAlpha = max(alphaX, alphaY);
        gridAlpha *= 0.01; // Very faint
    }
    
    // Axes Logic (X=0, Y=0)
    float axisAlpha = 0.0;
    // Axis width: 1.5 pixels
    vec2 axisWidth = 1.0 * derivative;
    
    if (abs(worldPos.x) < axisWidth.x || abs(worldPos.y) < axisWidth.y) {
         float axA = (abs(worldPos.x) < axisWidth.x) ? (1.0 - smoothstep(0.0, axisWidth.x, abs(worldPos.x))) : 0.0;
         float axB = (abs(worldPos.y) < axisWidth.y) ? (1.0 - smoothstep(0.0, axisWidth.y, abs(worldPos.y))) : 0.0;
         axisAlpha = max(axA, axB);
         axisAlpha *= 0.05; // Faint
    }
    
    float finalAlpha = max(gridAlpha, axisAlpha);
    
    if (finalAlpha <= 0.0) discard;
    
    outColor = vec4(1.0, 1.0, 1.0, finalAlpha);
}
