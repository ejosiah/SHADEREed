#version 460

uniform mat4 viewProjection;
uniform mat4 GeometryTransform;

layout(location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;



void main(){
	gl_Position = viewProjection * GeometryTransform * vec4(pos, 1);
}