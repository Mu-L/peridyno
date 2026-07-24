#pragma once
#include "Module/ComputeModule.h"
#include "Topology/TextureMesh.h"
#include "Primitive/Primitive3D.h"

// #define FRUSTUM_CULL_DEBUG

namespace dyno
{

	class ComputeFrustumCullTransform : public ComputeModule
	{
		DECLARE_CLASS(ComputeFrustumCullTransform)
	public:
		typedef typename dyno::TPlane3D<Real> Plane3D;

		ComputeFrustumCullTransform();
		~ComputeFrustumCullTransform() override;

		DEF_ARRAYLIST_IN(Transform3f, Transform, DeviceType::GPU, "Instance transforms");

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "TextureMesh whose shapes provide bounding volumes");

		DEF_ARRAY_IN(Plane3D, FrustumPlanes, DeviceType::GPU, "Six frustum planes in world space");

		DEF_ARRAYLIST_OUT(Transform3f, VisibleTransform, DeviceType::GPU, "Visible instance transforms");

	protected:
		void compute() override;
	};
}
