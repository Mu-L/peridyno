#include "GLPhotorealisticInstanceRender.h"
#include "ComputeFrustumCullTransform.h"
#include "Utility.h"

#include <glad/glad.h>

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

#include "ShaderStruct.h"

// 是否开启视锥剔除（设为 false 则退化为原来的仅 LOD 模式）
#define ENABLE_FRUSTUM_CULL false

namespace dyno
{
	IMPLEMENT_CLASS(GLPhotorealisticInstanceRender)

	GLPhotorealisticInstanceRender::GLPhotorealisticInstanceRender()
		: GLPhotorealisticRender()
	{

#ifdef CUDA_BACKEND
		mComputeLodTransform = std::make_shared<ComputeLodTransform>();
		this->inTextureMesh()->connect(mComputeLodTransform->inTextureMesh());
		this->inTransform()->connect(mComputeLodTransform->inTransform());

		// 创建视锥剔除模块，连接在 LOD 模块之前：
		// inTransform -> ComputeFrustumCull -> outVisibleTransform -> ComputeLodTransform -> outTransformLod*
		mComputeFrustumCull = std::make_shared<ComputeFrustumCullTransform>();
		this->inTextureMesh()->connect(mComputeFrustumCull->inTextureMesh());
		this->inTransform()->connect(mComputeFrustumCull->inTransform());
		// 将视锥剔除的可见输出连接到 LOD 模块的输入
		mComputeFrustumCull->outVisibleTransform()->connect(mComputeLodTransform->inTransform());
#endif

	}

	GLPhotorealisticInstanceRender::~GLPhotorealisticInstanceRender()
	{
	
	}

	std::string GLPhotorealisticInstanceRender::caption()
	{
		return "Photorealistic Instance Render";
	}

	bool GLPhotorealisticInstanceRender::initializeGL()
	{
		mXTransformBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mXTransformBufferLod1.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mXTransformBufferLod2.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		return GLPhotorealisticRender::initializeGL();
	}

	void GLPhotorealisticInstanceRender::releaseGL()
	{
		mXTransformBuffer.release();
		mXTransformBufferLod1.release();
		mXTransformBufferLod2.release();
		GLPhotorealisticRender::releaseGL();
	}

	void GLPhotorealisticInstanceRender::updateGL()
	{
		GLPhotorealisticRender::updateGL();
	}


	void GLPhotorealisticInstanceRender::updateImpl()
	{
#ifdef CUDA_BACKEND

		if (!this->inTextureMesh()->constDataPtr()->useLod())
		{
			auto transPtr = this->inTransform()->constDataPtr();

			mXTransformBuffer.load(transPtr->elements());
			if (this->inTransform()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(transPtr->index());
				mLists.assign(transPtr->lists());
				mNeedUpdateInstanceTransform = true;
			}
		}
#endif

		GLPhotorealisticRender::updateImpl();
	}

	glm::vec3 GetCameraPosition(const glm::mat4& viewMatrix)
	{
		glm::mat4 invView = glm::inverse(viewMatrix);
		return glm::vec3(invView[3]);
	}

	// ========================================================================
	// 从视图-投影矩阵提取 6 个视锥平面（世界空间，法线朝内）
	// ========================================================================
	// 视锥平面顺序约定（与 ComputeFrustumCullTransform 中的定义一致）：
	//   [0] Left   左平面
	//   [1] Right  右平面
	//   [2] Top    上平面
	//   [3] Bottom 下平面
	//   [4] Near   近平面
	//   [5] Far    远平面
	//
	// 算法原理：
	//   clipPos = VP * worldPos，其中 clipPos.w > 0
	//   规范化设备坐标 NDC = clipPos.xyz / clipPos.w
	//   视锥内的点满足 -1 <= NDC.x,y,z <= 1
	//   六个平面由 NDC 分量 = +/-1 定义，代入 clip 坐标推导平面方程
	//   最后变换到世界空间（乘以 VP 的逆的转置，即 (VP^-1)^T）
	//
	// 输入：
	//   view - 视图矩阵（world -> camera）
	//   proj - 投影矩阵（camera -> clip）
	// 输出：
	//   planes - 6 个视锥平面，存储在 outPlanes 数组中
	// ========================================================================
	void ExtractFrustumPlanes(const glm::mat4& view, const glm::mat4& proj, CArray<Plane3D>& outPlanes)
	{
		// VP = proj * view，将点从世界空间变换到裁剪空间
		// GLM 使用列主序，clip = VP * world 等价于 world^T * VP^T
		// 即 clip_j = (VP^T * world)_j = sum_k VP[k][j] * world_k
		// 列主序下 VP[j][k] 是第 j 列第 k 行，等价于 VP^T[k][j]
		glm::mat4 VP = proj * view;

		// GLM 列主序：VP[r][c] = 第 c 列第 r 行
		// clip.x = VP[0][0]*x + VP[1][0]*y + VP[2][0]*z + VP[3][0]*w
		// clip.y = VP[0][1]*x + VP[1][1]*y + VP[2][1]*z + VP[3][1]*w
		// clip.z = VP[0][2]*x + VP[1][2]*y + VP[2][2]*z + VP[3][2]*w
		// clip.w = VP[0][3]*x + VP[1][3]*y + VP[2][3]*z + VP[3][3]*w
		// NDC.x = clip.x / clip.w，要求 -1 <= NDC.x <= 1
		// 即 -clip.w <= clip.x <= clip.w
		//
		// 左平面 (clip.x <= clip.w)：clip.x + clip.w <= 0
		//   平面方程：(VP[0][0]+VP[3][0])*x + (VP[1][0]+VP[3][1])*y + ... = 0
		//   法线 N = (VP[0][0]+VP[3][0], VP[1][0]+VP[3][1], VP[2][0]+VP[3][2])
		// 右平面 (clip.x >= clip.w)：clip.x - clip.w >= 0
		//   N = (VP[0][0]-VP[3][0], VP[1][0]-VP[3][1], VP[2][0]-VP[3][2])
		// 上平面 (clip.y <= clip.w)：N = (VP[0][1]+VP[3][1], VP[1][1]+VP[3][1], VP[2][1]+VP[3][2])
		// 下平面 (clip.y >= clip.w)：N = (VP[0][1]-VP[3][1], VP[1][1]-VP[3][1], VP[2][1]-VP[3][2])
		//
		// 视锥近/远平面由 clip.z 决定（假设 z>0 朝向摄像机，z=near 时 NDC.z=-1，z=far 时 NDC.z=1）：
		// 近平面：N = (VP[0][2]+VP[3][2], VP[1][2]+VP[3][2], VP[2][2]+VP[3][2])，其中 (VP[3][2]+VP[3][3])/n < 0
		// 远平面：N = (VP[0][2]-VP[3][2], VP[1][2]-VP[3][2], VP[2][2]-VP[3][2])，其中 (VP[3][2]-VP[3][3])/f > 0
		//
		// 归一化后：plane.origin = plane.normal * (-d)，其中 d = -dot(normal, plane_point)

		glm::mat4 VP_T = glm::transpose(VP);

		outPlanes.resize(6);

		// [0] Left
		glm::vec4 left_n = VP_T[0] + VP_T[3];
		float left_len = sqrt(left_n.x * left_n.x + left_n.y * left_n.y + left_n.z * left_n.z);
		left_n /= left_len;
		outPlanes[0].normal = Vec3f(left_n.x, left_n.y, left_n.z);
		outPlanes[0].origin = outPlanes[0].normal * (-left_n.w / left_len);

		// [1] Right
		glm::vec4 right_n = VP_T[3] - VP_T[0];
		float right_len = sqrt(right_n.x * right_n.x + right_n.y * right_n.y + right_n.z * right_n.z);
		right_n /= right_len;
		outPlanes[1].normal = Vec3f(right_n.x, right_n.y, right_n.z);
		outPlanes[1].origin = outPlanes[1].normal * (-right_n.w / right_len);

		// [2] Top
		glm::vec4 top_n = VP_T[1] + VP_T[3];
		float top_len = sqrt(top_n.x * top_n.x + top_n.y * top_n.y + top_n.z * top_n.z);
		top_n /= top_len;
		outPlanes[2].normal = Vec3f(top_n.x, top_n.y, top_n.z);
		outPlanes[2].origin = outPlanes[2].normal * (-top_n.w / top_len);

		// [3] Bottom
		glm::vec4 bottom_n = VP_T[3] - VP_T[1];
		float bottom_len = sqrt(bottom_n.x * bottom_n.x + bottom_n.y * bottom_n.y + bottom_n.z * bottom_n.z);
		bottom_n /= bottom_len;
		outPlanes[3].normal = Vec3f(bottom_n.x, bottom_n.y, bottom_n.z);
		outPlanes[3].origin = outPlanes[3].normal * (-bottom_n.w / bottom_len);

		// [4] Near
		glm::vec4 near_n = VP_T[2] + VP_T[3];
		float near_len = sqrt(near_n.x * near_n.x + near_n.y * near_n.y + near_n.z * near_n.z);
		near_n /= near_len;
		outPlanes[4].normal = Vec3f(near_n.x, near_n.y, near_n.z);
		outPlanes[4].origin = outPlanes[4].normal * (-near_n.w / near_len);

		// [5] Far
		glm::vec4 far_n = VP_T[3] - VP_T[2];
		float far_len = sqrt(far_n.x * far_n.x + far_n.y * far_n.y + far_n.z * far_n.z);
		far_n /= far_len;
		outPlanes[5].normal = Vec3f(far_n.x, far_n.y, far_n.z);
		outPlanes[5].origin = outPlanes[5].normal * (-far_n.w / far_len);
	}

	void GLPhotorealisticInstanceRender::paintGL(const RenderParams& rparams)
	{
		RenderParams rp = rparams;
		rp.transforms.model = glm::mat4{ 1.0 };
		mRenderParamsUBlock.load((void*)&rp, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		glm::vec3 cameraPosition = GetCameraPosition(rp.transforms.view);
		Vec3f camP = Vec3f(cameraPosition.x, cameraPosition.y, cameraPosition.z);

#ifdef CUDA_BACKEND

		mShaderProgram->use();

		auto transPtr = this->inTransform()->constDataPtr();

		// 是否需要重新执行视锥剔除：摄像机或投影矩阵发生变化时
		bool frustumChanged =
			(cameraPosition.x != mLastCullCameraPos.x ||
			 cameraPosition.y != mLastCullCameraPos.y ||
			 cameraPosition.z != mLastCullCameraPos.z) ||
			(rp.transforms.view != mLastCullViewMat) ||
			(rp.transforms.proj != mLastCullProjMat);

		// 从视图-投影矩阵提取视锥平面（存储在 CPU 端 CArray 中）
		if (frustumChanged)
		{
			ExtractFrustumPlanes(rp.transforms.view, rp.transforms.proj, mFrustumPlanes);
			mLastCullCameraPos = cameraPosition;
			mLastCullViewMat = rp.transforms.view;
			mLastCullProjMat = rp.transforms.proj;
		}

		if(this->inTextureMesh()->constDataPtr()->useLod())
		{
			// ================================================================
			// 视锥剔除 + LOD 计算流程
			// ================================================================
			// 1. 提取视锥平面（如需要）
			// 2. 设置视锥平面到剔除模块，执行视锥剔除得到可见实例
			// 3. 将可见实例传给 LOD 模块，计算细节层次
			// 4. 渲染各 LOD 层级
			// ================================================================

			if (camP.norm() >= 0.0001) // && (mCamPosition != camP) || this->inTransform()->isModified()
			{
#if ENABLE_FRUSTUM_CULL
				// Step 1: 将视锥平面上传到 GPU，执行视锥剔除
				DArray<Plane3D> dPlanes;
				dPlanes.assign(mFrustumPlanes);
				mComputeFrustumCull->inFrustumPlanes()->assign(dPlanes);
				mComputeFrustumCull->update();
				dPlanes.clear();
#endif

				// Step 2: 执行 LOD 计算（输入来自视锥剔除的可见实例）
				mComputeLodTransform->inCameraPos()->setValue(mCamPosition);
				mComputeLodTransform->update();

#if ENABLE_FRUSTUM_CULL
				// Step 3: 加载视锥剔除后的可见实例（用于渲染）
				mXTransformBuffer.load(mComputeFrustumCull->outVisibleTransform()->constDataPtr()->elements());
#else
				// Step 3 (无视锥剔除): 加载原始实例
				mXTransformBuffer.load(mComputeLodTransform->outTransformLod0()->constDataPtr()->elements());
#endif

				mXTransformBufferLod1.load(mComputeLodTransform->outTransformLod1()->constDataPtr()->elements());
				mXTransformBufferLod2.load(mComputeLodTransform->outTransformLod2()->constDataPtr()->elements());

				mXTransformBuffer.updateGL();
				mXTransformBufferLod1.updateGL();
				mXTransformBufferLod2.updateGL();

				mCamPosition = Vec3f(cameraPosition.x, cameraPosition.y, cameraPosition.z);
			}
			else
			{
				// 摄像机在原点时，直接使用缓存数据
				mXTransformBuffer.load(mComputeLodTransform->outTransformLod0()->constDataPtr()->elements());
				mXTransformBufferLod1.load(mComputeLodTransform->outTransformLod1()->constDataPtr()->elements());
				mXTransformBufferLod2.load(mComputeLodTransform->outTransformLod2()->constDataPtr()->elements());

				mXTransformBuffer.updateGL();
				mXTransformBufferLod1.updateGL();
				mXTransformBufferLod2.updateGL();
			}

			// ================================================================
			// UpdateTransform & paintLOD
			// ================================================================
#if ENABLE_FRUSTUM_CULL
			// 视锥剔除后的可见实例作为渲染数据源
			auto& cullOutPtr = mComputeFrustumCull->outVisibleTransform()->constDataPtr();

			if (mComputeLodTransform->outTransformLod0()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(mComputeLodTransform->outTransformLod0()->constDataPtr()->index());
				mLists.assign(mComputeLodTransform->outTransformLod0()->constDataPtr()->lists());
				mNeedUpdateInstanceTransform = true;
			}

			paintLOD(rparams, 0);

			if (mComputeLodTransform->outTransformLod1()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(mComputeLodTransform->outTransformLod1()->constDataPtr()->index());
				mLists.assign(mComputeLodTransform->outTransformLod1()->constDataPtr()->lists());
				mNeedUpdateInstanceTransform = true;
			}
			paintLOD(rparams, 1);

			if (mComputeLodTransform->outTransformLod2()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(mComputeLodTransform->outTransformLod2()->constDataPtr()->index());
				mLists.assign(mComputeLodTransform->outTransformLod2()->constDataPtr()->lists());
				mNeedUpdateInstanceTransform = true;
			}
			paintLOD(rparams, 2);
#else
			// 无视锥剔除模式（原来的逻辑）
			if (mComputeLodTransform->outTransformLod0()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(mComputeLodTransform->outTransformLod0()->constDataPtr()->index());
				mLists.assign(mComputeLodTransform->outTransformLod0()->constDataPtr()->lists());
				mNeedUpdateInstanceTransform = true;
			}

			paintLOD(rparams, 0);

			if (mComputeLodTransform->outTransformLod1()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(mComputeLodTransform->outTransformLod1()->constDataPtr()->index());
				mLists.assign(mComputeLodTransform->outTransformLod1()->constDataPtr()->lists());
				mNeedUpdateInstanceTransform = true;
			}
			paintLOD(rparams, 1);

			if (mComputeLodTransform->outTransformLod2()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(mComputeLodTransform->outTransformLod2()->constDataPtr()->index());
				mLists.assign(mComputeLodTransform->outTransformLod2()->constDataPtr()->lists());
				mNeedUpdateInstanceTransform = true;
			}
			paintLOD(rparams, 2);
#endif

		}
		else
		{
			// ================================================================
			// 非 LOD 模式：仅应用视锥剔除后直接渲染
			// ================================================================
#if ENABLE_FRUSTUM_CULL
			if (frustumChanged || this->inTransform()->isModified())
			{
				DArray<Plane3D> dPlanes;
				dPlanes.assign(mFrustumPlanes);
				mComputeFrustumCull->inFrustumPlanes()->assign(dPlanes);
				mComputeFrustumCull->update();
				dPlanes.clear();
			}

			auto cullOutPtr = mComputeFrustumCull->outVisibleTransform()->constDataPtr();
			if (cullOutPtr && cullOutPtr->elementSize() > 0)
			{
				mXTransformBuffer.load(cullOutPtr->elements());
				mOffset.assign(cullOutPtr->index());
				mLists.assign(cullOutPtr->lists());
				mNeedUpdateInstanceTransform = true;
			}
#else
			// 无视锥剔除
			mXTransformBuffer.load(transPtr->elements());
			mXTransformBuffer.updateGL();
			if (this->inTransform()->isModified())
			{
				auto texMesh = this->inTextureMesh()->constDataPtr();
				mOffset.assign(transPtr->index());
				mLists.assign(transPtr->lists());
				mNeedUpdateInstanceTransform = true;
			}
#endif

			paintLOD(rparams, 0);
		}
#endif
	}


	void GLPhotorealisticInstanceRender::paintLOD(const RenderParams& rparams, int level)
	{
		auto& vertices = mTextureMesh.verticesLOD(level); 
		if (vertices.count() == 0) 
			return;

		auto& normals = mTextureMesh.normalsLOD(level);
		if (normals.count() == 0)
			return;

		auto& texCoords = mTextureMesh.texCoordsLOD(level);
		//if (texCoords.count() == 0)
		//	return;

		XBuffer<Vec3f>& tangent = level == 0 ? mTangent : (level == 1 ? mTangentLOD1 : mTangentLOD2);
		XBuffer<Vec3f>& bitangent = level == 0 ? mBitangent : (level == 1 ? mBitangentLOD1 : mBitangentLOD2);

		//if (tangent.count() == 0)
		//	return;

		//if (bitangent.count() == 0)
		//	return;

		// setup uniforms
		if (normals.count() > 0
			&& tangent.count() > 0
			&& bitangent.count() > 0
			&& normals.count() == tangent.count()
			&& normals.count() == bitangent.count())
		{
			mShaderProgram->setInt("uVertexNormal", 1);
			normals.bindBufferBase(9);
			tangent.bindBufferBase(12);
			bitangent.bindBufferBase(13);
		}
		else
			mShaderProgram->setInt("uVertexNormal", 0);


		mShaderProgram->setInt("uInstanced", 1);
		//Reset the model transform
		RenderParams rp = rparams;
		rp.transforms.model = glm::mat4{ 1.0 };
		mRenderParamsUBlock.load((void*)&rp, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		vertices.bindBufferBase(8);
		texCoords.bindBufferBase(10);

		auto& shapes = mTextureMesh.shapesLOD(level);
		for (int i = 0; i < shapes.size(); i++)
		{
			auto shape = shapes[i];
			auto mtl = shape->material;

			// material 
			if (mtl != nullptr)
			{
				// material 
				{

					PBRMaterial pbr;
					auto color = this->varBaseColor()->getValue();

					pbr.color = { mtl->baseColor.x, mtl->baseColor.y, mtl->baseColor.z };
					pbr.metallic = mtl->metallic;
					pbr.roughness = mtl->roughness;
					pbr.alpha = mtl->alpha;
					pbr.EmissiveIntensity = mtl->emissiveIntensity;

					if (mtl->texORM.isValid())
						pbr.useAOTex = 1;
					if (mtl->texORM.isValid())
					{
						pbr.useRoughnessTex = 1;
						pbr.useMetallicTex = 1;
					}
					else
					{
						pbr.useRoughnessTex = 0;
						pbr.useMetallicTex = 0;
					}
					if (mtl->texEmissiveColor.isValid())
						pbr.useEmissiveTex = 1;
					else
						pbr.useEmissiveTex = 0;

					mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
					mPBRMaterialUBlock.bindBufferBase(1);
				}

				// bind textures 
				{
					// reset 
					glActiveTexture(GL_TEXTURE10);		// color
					glBindTexture(GL_TEXTURE_2D, 0);
					glActiveTexture(GL_TEXTURE11);		// bump map
					glBindTexture(GL_TEXTURE_2D, 0);
					glActiveTexture(GL_TEXTURE12);		// bump map
					glBindTexture(GL_TEXTURE_2D, 0);

					if (mtl->texColor.isValid()) {
						mShaderProgram->setInt("uColorMode", 2);
						mtl->texColor.bind(GL_TEXTURE10);
					}
					else {
						mShaderProgram->setInt("uColorMode", 1);
					}

					if (mtl->texBump.isValid()) {
						mtl->texBump.bind(GL_TEXTURE11);
						mShaderProgram->setFloat("uBumpScale", mtl->bumpScale);
					}
					if (mtl->texORM.isValid())
					{
						mtl->texORM.bind(GL_TEXTURE12);
					}
					if (mtl->texEmissiveColor.isValid())
					{
						mtl->texEmissiveColor.bind(GL_TEXTURE13);
					}
				}
			}
			else
			{
				// material 
				{
					PBRMaterial pbr;

					auto color = this->varBaseColor()->getValue();
					pbr.color = { color.r, color.g, color.b };
					pbr.metallic = this->varMetallic()->getValue();
					pbr.roughness = this->varRoughness()->getValue();
					pbr.alpha = this->varAlpha()->getValue();


					mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
					mPBRMaterialUBlock.bindBufferBase(1);
				}

				mShaderProgram->setInt("uColorMode", 1);
			}


			int numTriangles = shape->glVertexIndex.count();

			mVAO.bind();

			// setup VAO binding...
			{
				// vertex index
				shape->glVertexIndex.bind();
				glEnableVertexAttribArray(0);
				glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);

				if (shape->glNormalIndex.count() == numTriangles) {
					shape->glNormalIndex.bind();
					glEnableVertexAttribArray(1);
					glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
				}
				else
				{
					glDisableVertexAttribArray(1);
					glVertexAttribI4i(1, -1, -1, -1, -1);
				}

				if (shape->glTexCoordIndex.count() == numTriangles) {
					shape->glTexCoordIndex.bind();
					glEnableVertexAttribArray(2);
					glVertexAttribIPointer(2, 1, GL_INT, sizeof(int), (void*)0);
				}
				else
				{
					glDisableVertexAttribArray(2);
					glVertexAttribI4i(2, -1, -1, -1, -1);
				}

			}
			if (i < mOffset.size())
			{
				uint offset_i = sizeof(Transform3f) * mOffset[i];
				if (getLodTransformBuffer(level).count() == 0)
					continue;

				mVAO.bindVertexBuffer(&getLodTransformBuffer(level), 3, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 0, 1);
				// bind the scale vector
				mVAO.bindVertexBuffer(&getLodTransformBuffer(level), 4, 3, GL_FLOAT, sizeof(Transform3f), offset_i + sizeof(Vec3f), 1);
				// bind the rotation matrix
				mVAO.bindVertexBuffer(&getLodTransformBuffer(level), 5, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 2 * sizeof(Vec3f), 1);
				mVAO.bindVertexBuffer(&getLodTransformBuffer(level), 6, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 3 * sizeof(Vec3f), 1);
				mVAO.bindVertexBuffer(&getLodTransformBuffer(level), 7, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 4 * sizeof(Vec3f), 1);
				mVAO.bind();
				glDrawArraysInstanced(GL_TRIANGLES, 0, numTriangles * 3, mLists[i].size());

			}
			else
			{
				printf("GLPhotorealisticInstanceRender::inTransform Is Error !!!!!!\n");
			}
			mVAO.unbind();

		}

	}

	template<typename Transform3f, typename uint>
	__global__ void BuildElement2ListIndex(
		DArray<Transform3f> inTransElements,
		DArray<uint> inTransIndex,
		DArray<uint> elementListIndex
	)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= inTransElements.size()) return;
		uint index = 0;

		for (uint ListIndex = 0; ListIndex < inTransIndex.size(); ListIndex++)
		{
			index = inTransIndex[ListIndex];
			if (eId < index) 
			{
				elementListIndex[eId] = ListIndex - 1;

				return;
			}
		}
		elementListIndex[eId] = inTransIndex.size() - 1;
	}

	template<typename Transform3f, typename Vec3f, typename uint>
	__global__ void UpdateLODTransformSize(
		DArray<Transform3f> inTransElements,
		DArray<uint> elementListIndex,
		DArray<uint> sizeLod0,
		DArray<uint> sizeLod1,
		DArray<uint> sizeLod2,
		DArray<uint> elementInLod0,
		DArray<uint> elementInLod1,
		DArray<uint> elementInLod2,
		Vec3f cameraPos,
		float distanceLod1,
		float distanceLod2
	)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= inTransElements.size()) return;
		
		float d = abs((inTransElements[eId].translation() - cameraPos).norm());

		if (d < distanceLod1) 
		{
			atomicAdd(&sizeLod0[elementListIndex[eId]], 1);

			elementInLod0[eId] = 1;
			elementInLod1[eId] = 0;
			elementInLod2[eId] = 0;
		}
		else if (d >= distanceLod1 && d < distanceLod2) 
		{
			atomicAdd(&sizeLod1[elementListIndex[eId]], 1);

			elementInLod1[eId] = 1;
			elementInLod0[eId] = 0;
			elementInLod2[eId] = 0;
		}
		else if (d >= distanceLod2) 
		{
			atomicAdd(&sizeLod2[elementListIndex[eId]], 1);

			elementInLod0[eId] = 0;
			elementInLod1[eId] = 0;
			elementInLod2[eId] = 1;
		}
	}


	template<typename Transform3f, typename uint>
	__global__ void UpdateLODTransform(
		DArray<uint> elementInLod,
		DArray<Transform3f> inTransElements,
		DArrayList<Transform3f> lodElements,
		DArray<uint> elementListIndex,
		int level
	)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= elementInLod.size()) return;
		if (level == 2) 
		{
			//printf("ID : %d - %d - %d\n",eId, elementInLod[eId], elementListIndex[eId]);
		}
		if (elementInLod[eId] == 1)
		{
			uint listIndex = elementListIndex[eId];
			lodElements[listIndex].atomicInsert(inTransElements[eId]);

		}
	}

	IMPLEMENT_CLASS(ComputeLodTransform)

	ComputeLodTransform::ComputeLodTransform() 
	{
		this->outTransformLod0()->allocate();
		this->outTransformLod1()->allocate();
		this->outTransformLod2()->allocate();
	}
	
	ComputeLodTransform::~ComputeLodTransform() 
	{
	
	}

	void ComputeLodTransform::compute()
	{
		this->outTransformLod0()->getDataPtr()->clear();
		this->outTransformLod1()->getDataPtr()->clear();
		this->outTransformLod2()->getDataPtr()->clear();

		auto transPtr = this->inTransform()->constDataPtr();
		DArray<uint> element2ListIndex;
		element2ListIndex.resize(transPtr->elementSize());
		element2ListIndex.reset();

		auto& elements = transPtr->elements();
		auto& indexs = transPtr->index();

		cuExecute(transPtr->elementSize(),
			BuildElement2ListIndex,
			elements,
			indexs,
			element2ListIndex
		);


		DArray<uint> sizeLod0;
		DArray<uint> sizeLod1;
		DArray<uint> sizeLod2;
		sizeLod0.resize(transPtr->size());
		sizeLod1.resize(transPtr->size());
		sizeLod2.resize(transPtr->size());

		sizeLod0.reset();
		sizeLod1.reset();
		sizeLod2.reset();

		DArray<uint> elementInLod0;
		DArray<uint> elementInLod1;
		DArray<uint> elementInLod2;
		elementInLod0.resize(transPtr->elementSize());
		elementInLod1.resize(transPtr->elementSize());
		elementInLod2.resize(transPtr->elementSize());

		auto camPos = this->inCameraPos()->getValue();
		cuExecute(transPtr->elementSize(),
			UpdateLODTransformSize,
			transPtr->elements(),
			element2ListIndex,
			sizeLod0,
			sizeLod1,
			sizeLod2,
			elementInLod0,
			elementInLod1,
			elementInLod2,
			this->inCameraPos()->getValue(),
			this->inTextureMesh()->constDataPtr()->getLodDistance(1),
			this->inTextureMesh()->constDataPtr()->getLodDistance(2)
		);


		this->outTransformLod0()->getDataPtr()->resize(sizeLod0);
		this->outTransformLod1()->getDataPtr()->resize(sizeLod1);
		this->outTransformLod2()->getDataPtr()->resize(sizeLod2);


		//printf("+++++++++++++++++++++++++++++++\nSize:%d,   %d,   %d\n+++++++++++++++++++++++++++++++\n",
		//	this->outTransformLod0()->getDataPtr()->elements().size(),
		//	this->outTransformLod1()->getDataPtr()->elements().size(),
		//	this->outTransformLod2()->getDataPtr()->elements().size());


		auto& lod0s = this->outTransformLod0()->getData();

		if (sizeLod0.size()) 
		{

			cuExecute(elementInLod0.size(),
				UpdateLODTransform,
				elementInLod0,
				transPtr->elements(),
				lod0s,
				element2ListIndex,
				0
			);
		}

		if (sizeLod1.size()) 
		{
			cuExecute(elementInLod1.size(),
				UpdateLODTransform,
				elementInLod1,
				transPtr->elements(),
				this->outTransformLod1()->getData(),
				element2ListIndex,
				1
			);
		}
		if (sizeLod2.size())
		{

			cuExecute(elementInLod2.size(),
				UpdateLODTransform,
				elementInLod2,
				transPtr->elements(),
				this->outTransformLod2()->getData(),
				element2ListIndex,
				2
			);
		}


		element2ListIndex.clear();
		sizeLod0.clear();
		sizeLod1.clear();
		sizeLod2.clear();
		elementInLod0.clear();
		elementInLod1.clear();
		elementInLod2.clear();
	}


}