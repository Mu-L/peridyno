// Weighted Blended OIT - composite pass
// Reads the accumulation and revealage buffers and blends the resolved
// transparent layer over the (already rendered) opaque color buffer.
#version 440

#extension GL_GOOGLE_include_directive: enable

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

layout(binding = 0) uniform sampler2D uAccum;
layout(binding = 1) uniform sampler2D uReveal;

void main(void)
{
	vec4 accum = texture(uAccum, texCoord);
	float reveal = texture(uReveal, texCoord).a;

	// avgColor is the STRAIGHT (un-premultiplied) weighted average color:
	//   accum.rgb = sum(color * alpha * weight)
	//   accum.a   = sum(alpha * weight)
	//   -> avgColor = color  (alpha cancels out)
	vec3 avgColor = accum.rgb / max(accum.a, 1e-4);

	// coverage from the revealage buffer (already scaled by MSAA coverage)
	float alpha = 1.0 - reveal;

	// The engine composites with PREMULTIPLIED alpha-over (GL_ONE, ONE_MINUS_SRC_ALPHA).
	// avgColor is straight, so premultiply by the coverage alpha here. This keeps
	// partial-coverage edges (object/background boundary) and flickering z-fighting
	// regions correctly attenuated instead of over-contributing -> no white halo.
	fragColor = vec4(avgColor * alpha, alpha);
}
