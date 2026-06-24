#include "ComputeFrustumCullTransform.h"
#include <cuda_runtime.h>

#include "Module/ComputeModule.h"
#include "Topology/TextureMesh.h"
#include "Primitive/Primitive3D.h"
#include <stdio.h>

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

	// ============================================================================
	// 核函数 1：BuildElement2ListIndex
	// ============================================================================
	// 功能：建立「元素全局索引 -> 所属列表索引」的映射数组
	//
	// 背景知识 - DArrayList 的存储结构：
	//   DArrayList 将多个 List 的元素连续存储在一个扁平数组中，
	//   并通过 index 数组记录每个 List 的结束位置（前缀和形式）。
	//   例如有 3 个 List，各含 2/3/1 个元素：
	//     elements: [e0, e1, e2, e3, e4, e5]
	//     index:    [2, 5, 6]   （List0 结束于 2，List1 结束于 5，List2 结束于 6）
	//   这种结构便于 GPU 并行访问元素，但无法直接知道某个元素属于哪个 List。
	//
	// 算法：
	//   对每个元素全局索引 eId，遍历 index 数组，
	//   找到第一个满足 eId < index[i] 的 i，则该元素属于 List (i==0 ? 0 : i-1)
	//
	// 输入：
	//   inTransElements - 扁平存储的所有实例变换元素
	//   inTransIndex    - 每个 List 的结束位置（前缀和）
	// 输出：
	//   elementListIndex - 元素全局索引 -> 所属列表索引 的映射
	// ============================================================================
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

	// ============================================================================
	// 核函数 2：MarkVisibleAndCount
	// ============================================================================
	// 功能：核心视锥剔除 —— 标记每个实例是否可见，并按列表统计可见数量
	//
	// 视锥剔除算法原理（包围球 vs 平面的有符号距离测试）：
	//
	//   视锥由 6 个平面围成，每个平面的法线指向视锥内部。
	//   对于一个包围球（球心 C，半径 R）和一个平面（点 P，法线 N）：
	//
	//                    N（法线朝内）
	//                    |
	//          外侧      |     内侧
	//                    |
	//       <---- dist ----
	//                    |
	//                    P（平面上一点）
	//
	//   球心到平面的有符号距离：dist = (C - P) . N
	//
	//   判定条件：
	//     dist >=  R   -> 球完全在平面内侧     -> 保留
	//     dist >  -R   -> 球与平面相交         -> 保留（保守）
	//     dist <= -R   -> 球完全在平面外侧     -> 剔除
	//
	//   包围球在所有 6 个平面上都满足 dist > -R，才认为在视锥内（可见）。
	//   这是一个保守测试：可能将部分实际不可见的物体误判为可见（假阳性），
	//   但绝不会将可见物体误判为不可见（无假阴性），保证渲染正确性。
	//
	// 世界空间包围球计算：
	//   每个 Shape 有一个预计算的 Shape 空间包围球（球心 shapeCenter，半径 shapeRadius），
	//   通过实例变换 T 将其变换到世界空间：
	//     worldCenter = T * shapeCenter       （平移 + 旋转）
	//     worldRadius = shapeRadius * maxScale  （取最大缩放分量，保守估计）
	//   使用最大缩放分量是保守策略——保证世界空间的球完全包住变换后的实际形状。
	//
	// 原子操作：
	//   由于多个线程可能同时增加同一个 List 的可见计数，使用 atomicAdd 保证计数正确。
	//
	// 输入：
	//   inTransElements    - 所有实例变换（扁平数组）
	//   elementListIndex   - 元素 -> 列表索引映射
	//   frustumPlanes      - 6 个视锥平面（世界空间，法线朝内）
	//   shapeSphereCenters - 每个 Shape 的包围球心（Shape 空间）
	//   shapeSphereRadii   - 每个 Shape 的包围球半径（Shape 空间）
	// 输出：
	//   visibleCount  - 每个 List 的可见实例数量
	//   elementVisible - 每个元素的可见性标记（1=可见，0=被剔除）
	// ============================================================================
	template<typename Transform3f, typename Vec3f, typename Plane3D, typename Real, typename uint>
	__global__ void MarkVisibleAndCount(
		DArray<Transform3f> inTransElements,
		DArray<uint> elementListIndex,
		DArray<Plane3D> frustumPlanes,
		DArray<Vec3f> shapeSphereCenters,
		DArray<Real> shapeSphereRadii,
		DArray<uint> visibleCount,
		DArray<uint> elementVisible)
	{
		int eId = threadIdx.x + blockIdx.x * blockDim.x;
		if (eId >= inTransElements.size()) return;

		// 第一步：找到该元素所属的 Shape / List 索引
		uint listIdx = elementListIndex[eId];

		// 第二步：获取该 Shape 的包围球（Shape 空间，已应用 boundingTransform）
		Vec3f shapeCenter = shapeSphereCenters[listIdx];
		Real shapeRadius = shapeSphereRadii[listIdx];

		// 第三步：计算世界空间的包围球
		Transform3f trans = inTransElements[eId];
		Vec3f worldCenter = trans * shapeCenter;  // 球心变换到世界空间

		// 半径取最大缩放分量，保守估计（宁可多画也不漏画）
		Vec3f s = trans.scale();
		Real maxScale = fmaxf(fmaxf(fabsf(s[0]), fabsf(s[1])), fabsf(s[2]));
		Real worldRadius = shapeRadius * maxScale;

#ifdef FRUSTUM_CULL_DEBUG
		// 调试打印：只打印前 3 个实例的详细计算过程
		if (eId < 3)
		{
			printf("[GPU MarkVisible] eId=%d listIdx=%u\n", eId, listIdx);
			printf("  shapeCenter=(%f,%f,%f) shapeRadius=%f\n",
				shapeCenter[0], shapeCenter[1], shapeCenter[2], shapeRadius);
			printf("  worldCenter=(%f,%f,%f) worldRadius=%f maxScale=%f\n",
				worldCenter[0], worldCenter[1], worldCenter[2], worldRadius, maxScale);
		}
#endif

		// 第四步：6 平面视锥测试
		bool inside = true;
		for (int i = 0; i < 6; ++i)
		{
			// 球心到平面的有符号距离 = (球心 - 平面上一点) . 平面法线
			Real dist = (worldCenter - frustumPlanes[i].origin).dot(frustumPlanes[i].normal);

#ifdef FRUSTUM_CULL_DEBUG
			if (eId < 3)
			{
				printf("  plane[%d] dist=%f origin=(%f,%f,%f) normal=(%f,%f,%f) %s\n",
					i, dist,
					frustumPlanes[i].origin[0], frustumPlanes[i].origin[1], frustumPlanes[i].origin[2],
					frustumPlanes[i].normal[0], frustumPlanes[i].normal[1], frustumPlanes[i].normal[2],
					dist < -worldRadius ? "CULLED" : "inside");
			}
#endif

			// 完全在该平面外侧 -> 立即剔除
			if (dist < -worldRadius)
			{
				inside = false;
				break;
			}
		}

		// 第五步：更新可见标记和计数
		if (inside)
		{
			elementVisible[eId] = 1u;
			atomicAdd(&visibleCount[listIdx], 1);  // 原子累加，线程安全
		}
		else
		{
			elementVisible[eId] = 0u;
		}

#ifdef FRUSTUM_CULL_DEBUG
		if (eId < 3)
		{
			printf("  => %s\n", inside ? "VISIBLE" : "CULLED");
		}
#endif
	}

	// ============================================================================
	// 核函数 3：FillVisibleTransforms
	// ============================================================================
	// 功能：将可见实例的变换原子插入输出 DArrayList 的对应列表中
	//
	// 算法：
	//   遍历所有元素，若 elementVisible[eId] == 1（可见），
	//   则通过 outTransList[listIndex].atomicInsert() 将变换插入对应列表。
	//
	// 为什么用 atomicInsert：
	//   输出列表已通过 resize() 预分配了空间（大小等于该 List 的可见数），
	//   但多个线程可能同时写入同一个 List，需要原子操作保证索引正确。
	//   atomicInsert 内部维护一个原子计数器，每次调用返回当前写入位置并自增。
	//
	// 输入：
	//   elementVisible  - 每个元素的可见性标记
	//   inTransElements - 所有实例变换（扁平数组）
	//   elementListIndex - 元素 -> 列表索引映射
	// 输出：
	//   outTransList - 输出的可见变换 DArrayList
	// ============================================================================
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

	// ============================================================================
	// compute() — 视锥剔除主函数
	// ============================================================================
	// 整体执行流程：
	//
	//   [CPU] 输入校验
	//      |
	//   [CPU] 为每个 Shape 预计算包围球（应用 boundingTransform），上传到 GPU
	//      |
	//   [GPU] 核函数 1：BuildElement2ListIndex  ← 建立元素到列表的索引映射
	//      |
	//   [GPU] 核函数 2：MarkVisibleAndCount     ← 核心视锥剔除 + 计数
	//      |
	//   [CPU] 按可见数量 resize 输出列表
	//      |
	//   [GPU] 核函数 3：FillVisibleTransforms   ← 填充可见变换
	//      |
	//   [CPU] 清理临时 GPU 内存
	// ============================================================================
	void ComputeFrustumCullTransform::compute()
	{
		// ---- 输入校验 ----
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

#ifdef FRUSTUM_CULL_DEBUG
		printf("============================================================\n");
		printf("[FrustumCull] === 输入数据 ===\n");
		printf("[FrustumCull] Shape 数量: %u, List 数量: %u, 总实例数: %u\n",
			shapeCount, listCount, elementCount);

		// 打印每个 List 的元素数量
		auto& indexs = transPtr->index();
		CArray<uint> h_index;
		h_index.assign(indexs);
		printf("[FrustumCull] 各 List 元素数量: ");
		for (uint i = 0; i < listCount; ++i)
		{
			uint start = (i == 0) ? 0 : h_index[i - 1];
			uint count = h_index[i] - start;
			printf("[%u]=%u ", i, count);
		}
		printf("\n");
		h_index.clear();

		// 打印 6 个视锥平面
		CArray<Plane3D> h_planes;
		h_planes.assign(*planesPtr);
		const char* planeNames[] = { "Left", "Right", "Top", "Bottom", "Near", "Far" };
		for (int i = 0; i < 6; ++i)
		{
			printf("[FrustumCull] Plane[%d] %-6s origin=(%f,%f,%f) normal=(%f,%f,%f)\n",
				i, planeNames[i],
				h_planes[i].origin[0], h_planes[i].origin[1], h_planes[i].origin[2],
				h_planes[i].normal[0], h_planes[i].normal[1], h_planes[i].normal[2]);
		}
		h_planes.clear();
#endif

		// ---- 第 0 步：在 CPU 端为每个 Shape 预计算包围球 ----
		// 从 Shape 的 boundingBox 推导局部包围球，再应用 boundingTransform
		// 得到 Shape 空间的包围球（球心 + 半径），然后上传到 GPU。
		CArray<Vec3f> h_shapeCenters(shapeCount);
		CArray<Real> h_shapeRadii(shapeCount);

		for (uint i = 0; i < shapeCount; ++i)
		{
			auto& shape = shapes[i];
			auto& bbox = shape->boundingBox;
			auto& bTrans = shape->boundingTransform;

			// 从 AABB 计算局部包围球：球心 = 对角线中点，半径 = 对角线一半
			Vec3f localCenter = (bbox.v0 + bbox.v1) * Real(0.5);
			Real localRadius = (bbox.v1 - bbox.v0).norm() * Real(0.5);

			// 应用 boundingTransform，得到 Shape 空间的包围球
			Vec3f shapeCenter = bTrans * localCenter;
			Vec3f s = bTrans.scale();
			Real maxScale = max(max(fabs(s[0]), fabs(s[1])), fabs(s[2]));
			Real shapeRadius = localRadius * maxScale;

			h_shapeCenters[i] = shapeCenter;
			h_shapeRadii[i] = shapeRadius;

#ifdef FRUSTUM_CULL_DEBUG
			printf("[FrustumCull] Shape[%u] bbox=[(%f,%f,%f)-(%f,%f,%f)] "
				"localCenter=(%f,%f,%f) localRadius=%f "
				"shapeCenter=(%f,%f,%f) shapeRadius=%f\n",
				i,
				bbox.v0[0], bbox.v0[1], bbox.v0[2],
				bbox.v1[0], bbox.v1[1], bbox.v1[2],
				localCenter[0], localCenter[1], localCenter[2], localRadius,
				shapeCenter[0], shapeCenter[1], shapeCenter[2], shapeRadius);
#endif
		}

		// 上传到 GPU
		DArray<Vec3f> d_shapeCenters;
		DArray<Real> d_shapeRadii;
		d_shapeCenters.assign(h_shapeCenters);
		d_shapeRadii.assign(h_shapeRadii);

		h_shapeCenters.clear();
		h_shapeRadii.clear();

		// 清空输出
		auto outDataPtr = this->outVisibleTransform()->getDataPtr();
		outDataPtr->clear();

		// ---- 第一步：建立元素 -> 列表索引映射 ----
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

		// ---- 第二步：视锥剔除 + 可见性标记 + 计数 ----
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
			d_shapeCenters,
			d_shapeRadii,
			visibleCount,
			elementVisible);

#ifdef FRUSTUM_CULL_DEBUG
		// 打印剔除统计结果
		CArray<uint> h_visibleCount;
		h_visibleCount.assign(visibleCount);

		// 在 CPU 端重新计算各 List 可见数，用于校验
		CArray<uint> h_elementVisible;
		h_elementVisible.assign(elementVisible);
		CArray<uint> h_elemListIndex;
		h_elemListIndex.assign(element2ListIndex);

		uint totalVisible = 0;
		printf("[FrustumCull] === 剔除结果 ===\n");
		for (uint i = 0; i < listCount; ++i)
		{
			uint cnt = 0;
			uint totalInList = 0;
			for (uint j = 0; j < elementCount; ++j)
			{
				if (h_elemListIndex[j] == i)
				{
					totalInList++;
					if (h_elementVisible[j] == 1) cnt++;
				}
			}
			totalVisible += cnt;
			printf("[FrustumCull] List[%u]: 可见=%u / 总数=%u (GPU计数=%u) %s\n",
				i, cnt, totalInList, h_visibleCount[i],
				cnt == h_visibleCount[i] ? "OK" : "MISMATCH!");
		}
		printf("[FrustumCull] 总计可见: %u / %u (%.2f%%)\n",
			totalVisible, elementCount,
			elementCount > 0 ? 100.0f * totalVisible / elementCount : 0.0f);

		// 打印前 5 个可见实例
		int printed = 0;
		printf("[FrustumCull] 前几个可见实例的 worldCenter:\n");
		CArray<Transform3f> h_elements;
		h_elements.assign(elements);
		CArray<Vec3f> h_shapeCenters2;
		h_shapeCenters2.assign(d_shapeCenters);
		CArray<Real> h_shapeRadii2;
		h_shapeRadii2.assign(d_shapeRadii);
		for (uint j = 0; j < elementCount && printed < 5; ++j)
		{
			if (h_elementVisible[j] == 1)
			{
				uint li = h_elemListIndex[j];
				Vec3f wc = h_elements[j] * h_shapeCenters2[li];
				Vec3f sc = h_elements[j].scale();
				Real ms = max(max(fabs(sc[0]), fabs(sc[1])), fabs(sc[2]));
				Real wr = h_shapeRadii2[li] * ms;
				printf("[FrustumCull]   eId=%u list=%u worldCenter=(%f,%f,%f) worldRadius=%f\n",
					j, li, wc[0], wc[1], wc[2], wr);
				printed++;
			}
		}
		h_elementVisible.clear();
		h_elemListIndex.clear();
		h_elements.clear();
		h_shapeCenters2.clear();
		h_shapeRadii2.clear();
		h_visibleCount.clear();
#endif

		// ---- 第三步：按可见数量预分配输出空间 ----
		// resize(visibleCount) 为每个 List 分配对应数量的空间，
		// 之后核函数 3 用 atomicInsert 将数据填入
		outDataPtr->resize(visibleCount);

		auto& outList = this->outVisibleTransform()->getData();

		cuExecute(elementCount,
			FillVisibleTransforms,
			elementVisible,
			elements,
			outList,
			element2ListIndex);

#ifdef FRUSTUM_CULL_DEBUG
		// 校验输出
		auto outPtr = this->outVisibleTransform()->constDataPtr();
		uint outElementCount = (uint)outPtr->elementSize();
		printf("[FrustumCull] === 输出校验 ===\n");
		printf("[FrustumCull] 输出总元素数: %u\n", outElementCount);
		CArray<uint> h_vc;
		h_vc.assign(visibleCount);
		for (uint i = 0; i < listCount; ++i)
		{
			uint outCnt = (uint)outPtr->getListSize(i);
			printf("[FrustumCull] List[%u] 输出大小=%u (预期=%u) %s\n",
				i, outCnt, h_vc[i],
				outCnt == h_vc[i] ? "OK" : "MISMATCH!");
		}
		h_vc.clear();
		printf("[FrustumCull] ============================================================\n");
#endif

		// ---- 清理临时 GPU 内存 ----
		element2ListIndex.clear();
		visibleCount.clear();
		elementVisible.clear();
		d_shapeCenters.clear();
		d_shapeRadii.clear();
	}
}
