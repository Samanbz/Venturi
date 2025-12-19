#version 450

layout(location =  0) in float inX;
layout(location =  1) in float inY;

void main() {
    gl_PointSize = 2.0;
    gl_Position = vec4(inX, inY, 0.0, 1.0);
}