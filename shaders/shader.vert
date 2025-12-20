#version 450

layout(location =  0) in float inX;
layout(location =  1) in float inY;

layout(push_constant) uniform PushConstants {
    mat4 projection;
} push;

void main() {
    gl_PointSize = 1.0;
    gl_Position = push.projection * vec4(inX, inY, 0.0, 1.0);
}