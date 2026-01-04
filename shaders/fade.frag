#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float fadeRate;
} push;

void main() {
    // Fade factor.
    // We use Reverse Subtraction: Dest = Dest - Src.
    // We subtract a small amount from both Color and Alpha.
    // This ensures it fades to black/transparent and doesn't get stuck.
    outColor = vec4(push.fadeRate, push.fadeRate, push.fadeRate, push.fadeRate);
}
