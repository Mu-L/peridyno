#include "GLPhotorealisticInstanceRender.h"
#include "ComputeFrustumCullTransform.h"
#include "Utility.h"

#include <glad/glad.h>

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

#include "ShaderStruct.h"

#define ENABLE_FRUSTUM_CULL true

namespace dyno
{
	IMPLEMENT_CLASS(GLPhotorealisticInstanceRender)

	GLPhotorealisticInstanceRender::GLPhotorealisticInstanceRender()
		: GLPhotorealisticRender()
	{

#ifdef CUDA_BACKEND
		mComputeLodTransform = std::make_shared<ComputeLodTransform>();
		this->inTextureMesh()->connect(mComputeLodTransform->inTextureMesh());

		mComputeFrustumCull = std::make_shared<ComputeFrustumCullTransform>();
		this->inTextureMesh()->connect(mComputeFrustumCull->inTextureMesh());
		this->inTransform()->connect(mComputeFrustumCull->inTransform());
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

		//if (!this->inTextureMesh()->constDataPtr()->useLod())
		//{
		//	auto transPtr = this->inTransform()->constDataPtr();

		//	mXTransformBuffer.load(transPtr->elements());
		//	if (this->inTransform()->isModified())
		//	{
		//		auto texMesh = this->inTextureMesh()->constDataPtr();
		//		mOffset.assign(transPtr->index());
		//		mLists.assign(transPtr->lists());
		//		mNeedUpdateInstanceTransform = true;
		//	}
		//}
#endif

		GLPhotorealisticRender::updateImpl();
	}

	glm::vec3 GetCameraPosition(const glm::mat4& viewMatrix)
	{
		glm::mat4 invView = glm::inverse(viewMatrix);
		return glm::vec3(invView[3]);
	}

	void ExtractFrustumPlanes(const glm::mat4& view, const glm::mat4& proj, CArray<Plane3D>& outPlanes)
	{
		outPlanes.resize(6);

		glm::mat4 invVP = glm::inverse(proj * view);

		glm::vec3 NDC_corners[8] = {
			glm::vec3(-1, -1, -1),
			glm::vec3( 1, -1, -1),
			glm::vec3(-1,  1, -1),
			glm::vec3( 1,  1, -1),
			glm::vec3(-1, -1,  1),
			glm::vec3( 1, -1,  1),
			glm::vec3(-1,  1,  1),
			glm::vec3( 1,  1,  1),
		};

		glm::vec3 corners[8];
		for (int i = 0; i < 8; ++i)
		{
			glm::vec4 p = invVP * glm::vec4(NDC_corners[i], 1.0f);
			corners[i] = glm::vec3(p) / p.w;
		}

		auto makePlane = [](glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 inwardPoint) -> Plane3D
		{
			glm::vec3 e1 = p1 - p0;
			glm::vec3 e2 = p2 - p0;
			glm::vec3 n = glm::normalize(glm::cross(e1, e2));
			if (glm::dot(inwardPoint - p0, n) < 0)
				n = -n;
			Plane3D plane;
			plane.normal = Vec3f(n.x, n.y, n.z);
			float d = -glm::dot(n, p0);
			plane.origin = Vec3f(n.x * (-d), n.y * (-d), n.z * (-d));
			return plane;
		};

		glm::vec3 center = (corners[0] + corners[7]) * 0.5f;

		outPlanes[0] = makePlane(corners[0], corners[2], corners[4], center);
		outPlanes[1] = makePlane(corners[1], corners[3], corners[5], center);
		outPlanes[2] = makePlane(corners[2], corners[3], corners[6], center);
		outPlanes[3] = makePlane(corners[0], corners[1], corners[4], center);
		outPlanes[4] = makePlane(corners[0], corners[1], corners[2], center);
		outPlanes[5] = makePlane(corners[4], corners[5], corners[6], center);
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

		bool frustumChanged =
			(cameraPosition.x != mLastCullCameraPos.x ||
			 cameraPosition.y != mLastCullCameraPos.y ||
			 cameraPosition.z != mLastCullCameraPos.z) ||
			(rp.transforms.view != mLastCullViewMat) ||
			(rp.transforms.proj != mLastCullProjMat);

		if (frustumChanged)
		{
			ExtractFrustumPlanes(rp.transforms.view, rp.transforms.proj, mFrustumPlanes);
			mLastCullCameraPos = cameraPosition;
			mLastCullViewMat = rp.transforms.view;
			mLastCullProjMat = rp.transforms.proj;
		}

		if(this->inTextureMesh()->constDataPtr()->useLod())
		{
			if (camP.norm() >= 0.0001)
			{
#if ENABLE_FRUSTUM_CULL
				DArray<Plane3D> dPlanes;
				dPlanes.assign(mFrustumPlanes);
				mComputeFrustumCull->inFrustumPlanes()->assign(dPlanes);
				mComputeFrustumCull->update();
				dPlanes.clear();
#endif

				mComputeLodTransform->inCameraPos()->setValue(mCamPosition);
				mComputeLodTransform->update();

				mXTransformBuffer.load(mComputeLodTransform->outTransformLod0()->constDataPtr()->elements());
				mXTransformBufferLod1.load(mComputeLodTransform->outTransformLod1()->constDataPtr()->elements());
				mXTransformBufferLod2.load(mComputeLodTransform->outTransformLod2()->constDataPtr()->elements());

				mXTransformBuffer.updateGL();
				mXTransformBufferLod1.updateGL();
				mXTransformBufferLod2.updateGL();

				mCamPosition = Vec3f(cameraPosition.x, cameraPosition.y, cameraPosition.z);
			}
			else
			{
				mXTransformBuffer.load(mComputeLodTransform->outTransformLod0()->constDataPtr()->elements());
				mXTransformBufferLod1.load(mComputeLodTransform->outTransformLod1()->constDataPtr()->elements());
				mXTransformBufferLod2.load(mComputeLodTransform->outTransformLod2()->constDataPtr()->elements());

				mXTransformBuffer.updateGL();
				mXTransformBufferLod1.updateGL();
				mXTransformBufferLod2.updateGL();
			}

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

		}
		else 
		{
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
				mXTransformBuffer.updateGL();
				mOffset.assign(cullOutPtr->index());
				mLists.assign(cullOutPtr->lists());
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
		//if (normals.count() == 0)
		//	return;

		auto& texCoords = mTextureMesh.texCoordsLOD(level);

		bool useInputNormal = false;

		if (normals.count() > 0) 
		{
			XBuffer<Vec3f>& tangent = level == 0 ? mTangent : (level == 1 ? mTangentLOD1 : mTangentLOD2);
			XBuffer<Vec3f>& bitangent = level == 0 ? mBitangent : (level == 1 ? mBitangentLOD1 : mBitangentLOD2);

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
				useInputNormal = true;
			}
		}

		if(!useInputNormal)
			mShaderProgram->setInt("uVertexNormal", 0);


		mShaderProgram->setInt("uInstanced", 1);
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

			if (mtl != nullptr)
			{
				PBRMaterial pbr;
				auto color = this->varBaseColor()->getValue();

				pbr.color = { mtl->baseColor.x, mtl->baseColor.y, mtl->baseColor.z };
				pbr.metallic = mtl->metallic;
				pbr.roughness = mtl->roughness;
				pbr.alpha = this->varUseGlobalAlpha()->getValue() ? this->varAlpha()->getValue() : mtl->alpha;
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

				glActiveTexture(GL_TEXTURE10);
				glBindTexture(GL_TEXTURE_2D, 0);
				glActiveTexture(GL_TEXTURE11);
				glBindTexture(GL_TEXTURE_2D, 0);
				glActiveTexture(GL_TEXTURE12);
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
			else
			{
				PBRMaterial pbr;

				auto color = this->varBaseColor()->getValue();
				pbr.color = { color.r, color.g, color.b };
				pbr.metallic = this->varMetallic()->getValue();
				pbr.roughness = this->varRoughness()->getValue();
				pbr.alpha = this->varUseGlobalAlpha()->getValue() ? this->varAlpha()->getValue() : this->varAlpha()->getValue();

				mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
				mPBRMaterialUBlock.bindBufferBase(1);

				mShaderProgram->setInt("uColorMode", 1);
			}


			int numTriangles = shape->glVertexIndex.count();

			mVAO.bind();

			{
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
				mVAO.bindVertexBuffer(&getLodTransformBuffer(level), 4, 3, GL_FLOAT, sizeof(Transform3f), offset_i + sizeof(Vec3f), 1);
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