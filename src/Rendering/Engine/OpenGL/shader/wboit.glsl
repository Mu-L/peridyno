
/*
* Weighted Blended Order-Independent Transparency (WBOIT)
* McGuire & Bavoil, "Weighted Blended Order-Independent Transparency" (2013)
*
* Shared helper used by the transparent surface / point / car shaders.
* It only provides the depth/opacity based weight function; the actual
* accumulation / revealage outputs are declared by each fragment shader.
*/

#ifndef WBOIT_GLSL
#define WBOIT_GLSL

// z: normalized device depth in [0,1] (gl_FragCoord.z)
// alpha: fragment opacity in [0,1]
float wboitWeight(float z, float alpha)
{
	float a = clamp(alpha, 0.0, 1.0);
	float w = clamp(
		pow(min(1.0, a * 10.0) + 0.01, 3.0) *
		1e8 *
		pow(1.0 - z, 3.0),
		1e-2, 3e3);
	return w;
}

#endif
