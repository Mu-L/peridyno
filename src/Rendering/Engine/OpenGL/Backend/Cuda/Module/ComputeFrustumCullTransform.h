#pragma once
#include "Module/ComputeModule.h"
#include "Topology/TextureMesh.h"
#include "Primitive/Primitive3D.h"

// 去掉下面一行的注释，即可开启视锥剔除的调试打印输出
// 开启后会打印：输入数据（视锥平面、Shape包围球、实例变换）、
// 中间过程（元素到列表的索引映射、每个元素的可见性标记）、
// 输出结果（各列表可见数量、输出列表大小校验）
// 注释掉则完全不产生调试代码，零性能开销
// #define FRUSTUM_CULL_DEBUG

namespace dyno
{
	/**
	 * @brief 摄像机视锥剔除模块（用于实例化渲染）
	 *
	 * 功能概述：
	 *   根据摄像机的 6 个视锥平面，对一组实例化渲染的 Transform 进行视锥剔除
	 *   （Frustum Culling），仅输出位于视锥内（或与视锥相交）的实例变换，
	 *   以减少后续渲染的 draw call 数量。
	 *
	 * 数据对应关系（关键设计）：
	 *   - 输入 inTransform (DArrayList<Transform3f>) 的第 i 个 List
	 *     对应 TextureMesh 的第 i 个 Shape
	 *   - 每个 Shape 有自己的 boundingBox 和 boundingTransform
	 *   - List 中的每个元素表示该 Shape 的一个实例渲染位置
	 *
	 * 剔除算法原理：
	 *   采用「包围球 + 6 平面有符号距离测试」：
	 *   1. 对每个 Shape，从其 boundingBox 推导局部包围球（球心 + 半径）
	 *   2. 应用 Shape 的 boundingTransform，得到 Shape 空间的包围球
	 *   3. 对每个实例，将 Shape 包围球通过实例变换变换到世界空间：
	 *        worldCenter = instanceTransform * shapeCenter
	 *        worldRadius = shapeRadius * max(|instanceScale|)   （保守估计）
	 *   4. 对 6 个视锥平面依次做有符号距离测试：
	 *        dist = (worldCenter - plane.origin) . plane.normal
	 *      若 dist < -worldRadius，则包围球完全在平面外侧，实例被剔除
	 *      （视锥平面法线朝内，点在平面外侧时有符号距离为负）
	 *   5. 只有包围球在所有 6 个平面上都满足 dist > -radius，才认为可见
	 *
	 * GPU 并行执行流程（三个 Kernel）：
	 *   Step 1: BuildElement2ListIndex
	 *           建立「元素全局索引 -> 所属列表索引」的映射
	 *           （DArrayList 将元素连续存储，需要反向查找归属）
	 *   Step 2: MarkVisibleAndCount
	 *           并行执行视锥测试，标记每个元素是否可见，并用原子操作按列表统计可见数
	 *   Step 3: FillVisibleTransforms
	 *           根据可见标记，用原子插入将可见变换填入输出 DArrayList
	 */
	class ComputeFrustumCullTransform : public ComputeModule
	{
		DECLARE_CLASS(ComputeFrustumCullTransform)
	public:
		typedef typename dyno::TPlane3D<Real> Plane3D;

		ComputeFrustumCullTransform();
		~ComputeFrustumCullTransform() override;

		/**
		 * @brief 输入：实例变换列表（DArrayList）
		 *
		 * DArrayList 结构说明：
		 *   - 内部由多个 List 组成，第 i 个 List 对应 TextureMesh 的第 i 个 Shape
		 *   - 每个 List 包含若干个 Transform3f，表示该 Shape 的多个实例位置
		 *   - 物理上所有 List 的元素连续存储在扁平数组中，通过 index 数组划分边界
		 */
		DEF_ARRAYLIST_IN(Transform3f, Transform, DeviceType::GPU, "Instance transforms");

		/**
		 * @brief 输入：纹理网格，提供每个 Shape 的包围体
		 *
		 * 每个 Shape 提供：
		 *   - boundingBox: 局部空间的轴对齐包围盒
		 *   - boundingTransform: 从局部空间到 Shape 空间的变换
		 */
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "TextureMesh whose shapes provide bounding volumes");

		/**
		 * @brief 输入：6 个视锥平面（世界空间，法线朝内）
		 *
		 * 顺序约定（对应标准 OpenGL 视锥平面）：
		 *   [0] Left   左平面
		 *   [1] Right  右平面
		 *   [2] Top    上平面
		 *   [3] Bottom 下平面
		 *   [4] Near   近平面
		 *   [5] Far    远平面
		 *
		 * 所有平面的 normal 指向视锥内部，即视锥内的点到平面的有符号距离为正
		 */
		DEF_ARRAY_IN(Plane3D, FrustumPlanes, DeviceType::GPU, "Six frustum planes in world space");

		/**
		 * @brief 输出：可见实例变换列表（DArrayList）
		 *
		 * 输出结构与输入一一对应：
		 *   第 i 个 List 包含第 i 个 Shape 的可见实例变换
		 */
		DEF_ARRAYLIST_OUT(Transform3f, VisibleTransform, DeviceType::GPU, "Visible instance transforms");

	protected:
		void compute() override;
	};
}
