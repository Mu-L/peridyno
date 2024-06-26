#version 440

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"
#include "shadow.glsl"

layout(location = 0) in VertexData
{
	vec3 position;
	vec3 normal;
};

layout(location = 0) out vec4 fragColor;

vec3 GetViewDir()
{
	// orthogonal projection
	if (uRenderParams.proj[3][3] == 1.0)
		return vec3(0, 0, 1);

	// perspective projection
	return normalize(-position);
}

vec3 reinhard_tonemap(vec3 v)
{
	return v / (1.0f + v);
}

vec3 gamma_correct(vec3 v)
{
	float gamma = 2.2;
	return pow(v, vec3(1.0 / gamma));
}

vec3 pbr();
void ColorPass(void)
{
	vec3 color = pbr();
	color = reinhard_tonemap(color);
	color = gamma_correct(color);
	fragColor.rgb = color;
	fragColor.a = 1.0;
}

void ShadowPass(void)
{
	float depth = gl_FragCoord.z;
	//depth = depth * 0.5 + 0.5;

	float moment1 = depth;
	float moment2 = depth * depth;

	// Adjusting moments (this is sort of bias per pixel) using partial derivative
	float dx = dFdx(depth);
	float dy = dFdy(depth);
	moment2 += 0.25 * (dx * dx + dy * dy);

	fragColor = vec4(moment1, moment2, 0.0, 0.0);
}

void main(void) { 
	if(uRenderParams.mode == 0){
		ColorPass();
	}else if(uRenderParams.mode == 1){
		ShadowPass();
	}else if(uRenderParams.mode == 2){
		discard;
	}
} 

// refer to https://learnopengl.com
const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / max(denom, 0.001); // prevent divide by zero for roughness=0.0 and NdotH=1.0
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 pbr()
{
	vec3 N = normalize(normal);
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;

	// calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
	// of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, uMtl.color, uMtl.metallic);

	// reflectance equation
	vec3 Lo = vec3(0.0);
	//for(int i = 0; i < 4; ++i) 
	{
		// calculate per-light radiance
		//vec3 L = normalize(lightPositions[i] - WorldPos);
		vec3 L = normalize(uRenderParams.direction.xyz);
		vec3 H = normalize(V + L);
		//float distance = length(lightPositions[i] - WorldPos);
		//float attenuation = 1.0 / (distance * distance);
		//vec3 radiance = lightColors[i] * attenuation;
		vec3 radiance = uRenderParams.intensity.rgb * uRenderParams.intensity.a;

		// Cook-Torrance BRDF
		float NDF = DistributionGGX(N, H, uMtl.roughness);
		float G = GeometrySmith(N, V, L, uMtl.roughness);
		vec3 F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

		vec3 nominator = NDF * G * F;
		float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
		vec3 specular = nominator / max(denominator, 0.001); // prevent divide by zero for NdotV=0.0 or NdotL=0.0

		// kS is equal to Fresnel
		vec3 kS = F;
		// for energy conservation, the diffuse and specular light can't
		// be above 1.0 (unless the surface emits light); to preserve this
		// relationship the diffuse component (kD) should equal 1.0 - kS.
		vec3 kD = vec3(1.0) - kS;
		// multiply kD by the inverse metalness such that only non-metals 
		// have diffuse lighting, or a linear blend if partly metal (pure metals
		// have no diffuse light).
		kD *= 1.0 - uMtl.metallic;

		// scale light by NdotL
		float NdotL = max(dot(N, L), 0.0);

		// add to outgoing radiance Lo
		//Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again

		Lo += GetShadowFactor(position) * (kD * uMtl.color / PI + specular) * radiance * NdotL;
	}

	vec3 ambient = uRenderParams.ambient.rgb * uRenderParams.ambient.a * uMtl.color;
	vec3 cameraLight = uRenderParams.camera.rgb * uRenderParams.camera.a * uMtl.color * abs(dotNV);
	return ambient + cameraLight + Lo;
}
