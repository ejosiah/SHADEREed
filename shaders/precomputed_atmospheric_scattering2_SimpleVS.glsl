#version 460

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;


uniform mat4 PV;
uniform mat4 M;
 
void main(){
	gl_Position = PV * M * vec4(pos, 0, 1);
}