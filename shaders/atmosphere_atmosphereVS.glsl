#version 460

uniform mat4 View;
uniform mat4 Projection;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;


layout(location = 0) out struct {
	vec3 viewDir;
	vec2 uv;
} vs_out;


void main(){
	vs_out.uv = uv;
	vec4 d = vec4(pos, 1, 1);
	d = inverse(Projection) * d;
	d /= d.w;
	vs_out.viewDir = (inverse(View) * normalize(d)).xyz;
	gl_Position = vec4(pos, 0, 1);
}
