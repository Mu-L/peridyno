#include "ComputeFrustumCullTransform.h"
#include <cuda_runtime.h>

#include "Module/ComputeModule.h"
#include "Topology/TextureMesh.h"
#include "Primitive/Primitive3D.h"
#include <stdio.h>

#define FRUSTUM_CULL_DEBUG

namespace dyno
{
	IMPLEMENT_CLASS(ComputeFrustumCullTransform)

	ComputeFrustumCullTransform::ComputeFrustumCullTransform()
	{
		this->outVisibleTransform()->allocate();
	}

	ComputeFrustumCullTransform::~ComputeFrustumCullTransform()
	{
	}

	template<typename Transform3f, typename uint>
	__global__ void BuildElement2ListIndex(
		DArray<Transform3f> inTransElements,
		DArray<uint> inTransIndex,
		DArray<uint> elementListIndex)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= inTransElements.size()) return;

		for (uint i = 0; i < inTransIndex.size(); ++i)
		{
			if (eId < inTransIndex[i])
			{
				elementListIndex[eId] = (i == 0) ? 0 : i - 1;
				return;
			}
		}
		elementListIndex[eId] = inTransIndex.size() - 1;
	}

	template<typename Transform3f, typename Plane3D, typename Real, typename uint>
	__global__ void MarkVisibleAndCount(
		DArray<Transform3f> inTransElements,
		DArray<uint> elementListIndex,
		DArray<Plane3D> frustumPlanes,
		DArray<Real> shapeSphereRadii,
		DArray<uint> visibleCount,
		DArray<uint> elementVisible)
	{
		int eId = threadIdx.x + blockIdx.x * blockDim.x;
		if (eId >= inTransElements.size()) return;

		uint listIdx = elementListIndex[eId];

		Real shapeRadius = shapeSphereRadii[listIdx];

		Transform3f trans = inTransElements[eId];
		auto worldCenter = trans.translation();

		auto s = trans.scale();
		Real maxScale = fmaxf(fmaxf(fabsf(s[0]), fabsf(s[1])), fabsf(s[2]));
		Real worldRadius = shapeRadius * maxScale;

		bool inside = true;
		for (int i = 0; i < 6; ++i)
		{
			Real dist = (worldCenter - frustumPlanes[i].origin).dot(frustumPlanes[i].normal);

			if (dist < -worldRadius)
			{
				inside = false;
				break;
			}
		}

		if (inside)
		{
			elementVisible[eId] = 1u;
			atomicAdd(&visibleCount[listIdx], 1);
		}
		else
		{
			elementVisible[eId] = 0u;
		}
	}

	template<typename Transform3f, typename uint>
	__global__ void FillVisibleTransforms(
		DArray<uint> elementVisible,
		DArray<Transform3f> inTransElements,
		DArrayList<Transform3f> outTransList,
		DArray<uint> elementListIndex)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= elementVisible.size()) return;

		if (elementVisible[eId] == 1)
		{
			uint listIndex = elementListIndex[eId];
			outTransList[listIndex].atomicInsert(inTransElements[eId]);
		}
	}

	void ComputeFrustumCullTransform::compute()
	{
		auto transPtr = this->inTransform()->constDataPtr();
		if (!transPtr || transPtr->elementSize() == 0)
			return;

		auto texMeshPtr = this->inTextureMesh()->constDataPtr();
		if (!texMeshPtr)
			return;

		auto planesPtr = this->inFrustumPlanes()->constDataPtr();
		if (!planesPtr || planesPtr->size() != 6)
			return;

		auto& shapes = texMeshPtr->shapes();
		uint shapeCount = (uint)shapes.size();
		if (shapeCount == 0)
			return;

		uint listCount = (uint)transPtr->size();
		if (listCount != shapeCount)
			return;

		uint elementCount = (uint)transPtr->elementSize();

		CArray<Real> h_shapeRadii(shapeCount);

		for (uint i = 0; i < shapeCount; ++i)
		{
			auto& shape = shapes[i];
			auto& bbox = shape->boundingBox;

			Real shapeRadius = (bbox.v1 - bbox.v0).norm() * Real(0.5);

			h_shapeRadii[i] = shapeRadius;
		}

		DArray<Real> d_shapeRadii;
		d_shapeRadii.assign(h_shapeRadii);

		h_shapeRadii.clear();

		auto outDataPtr = this->outVisibleTransform()->getDataPtr();
		outDataPtr->clear();

		DArray<uint> element2ListIndex;
		element2ListIndex.resize(elementCount);
		element2ListIndex.reset();

		auto& elements = transPtr->elements();
		auto& indexs2 = transPtr->index();

		cuExecute(elementCount,
			BuildElement2ListIndex,
			elements,
			indexs2,
			element2ListIndex);

		DArray<uint> visibleCount;
		visibleCount.resize(listCount);
		visibleCount.reset();

		DArray<uint> elementVisible;
		elementVisible.resize(elementCount);
		elementVisible.reset();

		cuExecute(elementCount,
			MarkVisibleAndCount,
			elements,
			element2ListIndex,
			*planesPtr,
			d_shapeRadii,
			visibleCount,
			elementVisible);
	
		outDataPtr->clear();
		outDataPtr->resize(visibleCount);
		
		auto& outList = this->outVisibleTransform()->getData();

		cuExecute(elementCount,
			FillVisibleTransforms,
			elementVisible,
			elements,
			outList,
			element2ListIndex);

#ifdef FRUSTUM_CULL_DEBUG
		auto outPtr = this->outVisibleTransform()->constDataPtr();
		uint outElementCount = (uint)outPtr->elementSize();
		printf("[FrustumCull] Out / In: %u / %u\n", outElementCount,this->inTransform()->constDataPtr()->elementSize());
		/*CArray<uint> h_vc;
		h_vc.assign(visibleCount);
		CArrayList<Transform3f> cL;
		cL.assign(this->outVisibleTransform()->getData());
		for (uint i = 0; i < listCount; ++i)
		{
			uint outCnt = cL.lists()[i].size();
			printf("[FrustumCull] List[%u] 输出大小=%u (预期=%u) %s\n",
				i, outCnt, h_vc[i],
				outCnt == h_vc[i] ? "OK" : "MISMATCH!");
		}
		h_vc.clear();
		printf("[FrustumCull] ============================================================\n");*/
#endif


		element2ListIndex.clear();
		visibleCount.clear();
		elementVisible.clear();
		d_shapeRadii.clear();
	}
}
