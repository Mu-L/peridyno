#pragma once
#include "GLPhotorealisticRender.h"
#include "ComputeFrustumCullTransform.h"

namespace dyno
{

	class ComputeLodTransform : public ComputeModule
	{
		DECLARE_CLASS(ComputeLodTransform)
	public:
		ComputeLodTransform();
		~ComputeLodTransform() override;

	public:

		DEF_VAR_IN(Vec3f, CameraPos, "");
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");

		DEF_ARRAYLIST_IN(Transform3f,Transform,DeviceType::GPU,"");

		DEF_ARRAYLIST_OUT(Transform3f,TransformLod0,DeviceType::GPU,"");
		DEF_ARRAYLIST_OUT(Transform3f,TransformLod1,DeviceType::GPU,"");
		DEF_ARRAYLIST_OUT(Transform3f,TransformLod2,DeviceType::GPU,"");

	protected:
		void compute() override;

	private:
	};




	class GLPhotorealisticInstanceRender : public GLPhotorealisticRender
	{
		DECLARE_CLASS(GLPhotorealisticInstanceRender)
	public:
		GLPhotorealisticInstanceRender();
		~GLPhotorealisticInstanceRender();
	public:
		virtual std::string caption() override;

		DEF_ARRAYLIST_IN(Transform3f, Transform, DeviceType::GPU, "");

	protected:
		void updateImpl() override;

		void paintGL(const RenderParams& rparams) override;

		void paintLOD(const RenderParams& rparams,int level);

		void updateGL() override;
		bool initializeGL() override;
		void releaseGL() override;

		XBuffer<Transform3f>& getLodTransformBuffer(int level) 
		{
			switch (level)
			{
			case 0:
				return mXTransformBuffer;
				break;
			case 1:
				return mXTransformBufferLod1;
				break;
			case 2:
				return mXTransformBufferLod2;
				break;
			default:
				break;
			}
		};


	private:
		CArray<uint> mOffset;
		CArray<List<Transform3f>> mLists;
			
		XBuffer<Transform3f> mXTransformBuffer;
		XBuffer<Transform3f> mXTransformBufferLod1;
		XBuffer<Transform3f> mXTransformBufferLod2;

		Vec3f mCamPosition = Vec3f(0);

		bool mNeedUpdateInstanceTransform = false;
		std::shared_ptr<ComputeLodTransform> mComputeLodTransform = NULL;
		std::shared_ptr<ComputeFrustumCullTransform> mComputeFrustumCull = NULL;
		CArray<Plane3D> mFrustumPlanes;
		glm::vec3 mLastCullCameraPos = glm::vec3(0.0f);
		glm::mat4 mLastCullProjMat = glm::mat4(1.0f);
		glm::mat4 mLastCullViewMat = glm::mat4(1.0f);
	};

};