#include "GLTextureMesh.h"
#include "CarMaterial.h"
#include "glad/glad.h"

namespace dyno
{
	/**
	 * GLMaterial
	 */
	GLMaterial::GLMaterial()
	{
	}

	GLMaterial::~GLMaterial()
	{
		release();
	}

	void GLMaterial::create()
	{
	}

	void GLMaterial::release()
	{
		texColor.release();
		texBump.release();
		texORM.release();
		texAlpha.release();
		texEmissiveColor.release();
	}

	void GLMaterial::updateGL()
	{
		texColor.updateGL();
		texBump.updateGL();
		texORM.updateGL();
		texAlpha.updateGL();
		texEmissiveColor.updateGL();

	}

	/**
	 * GLShape
	 */
	GLShape::GLShape()
	{
	}

	GLShape::~GLShape()
	{
		release();
	}

	void GLShape::create()
	{
		if (!mInitialized)
		{
			glVertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			glNormalIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			glTexCoordIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

			mInitialized = true;
		}
	}

	void GLShape::release()
	{
		glVertexIndex.release();
		glNormalIndex.release();
		glTexCoordIndex.release();
	}

	void GLShape::updateGL()
	{
		if (!mInitialized)
			create();

		glVertexIndex.updateGL();
		glNormalIndex.updateGL();
		glTexCoordIndex.updateGL();
		if (this->material != NULL)
			this->material->updateGL();

	}


	GLTextureMesh::GLTextureMesh()
	{
	}

	GLTextureMesh::~GLTextureMesh()
	{
	}

	void GLTextureMesh::create()
	{
		if (!mInitialized)
		{
			mVertices.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mNormal.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mTexCoord.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

			mVertices_LOD1.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mNormal_LOD1.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mTexCoord_LOD1.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

			mVertices_LOD2.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mNormal_LOD2.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mTexCoord_LOD2.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mInitialized = true;
		}
	}

	void GLTextureMesh::release()
	{
		mVertices.release();
		mNormal.release();
		mTexCoord.release();

		mVertices_LOD1.release();
		mNormal_LOD1.release();
		mTexCoord_LOD1.release();
		
		mVertices_LOD2.release();
		mNormal_LOD2.release();
		mTexCoord_LOD2.release();

		for (auto s : mShapes) {
			s->release();
		}

		for (auto s : mShapes_LOD1) {
			s->release();
		}

		for (auto s : mShapes_LOD2) {
			s->release();
		}

		mShapes.clear();
		mShapes_LOD1.clear();
		mShapes_LOD2.clear();
	}

#ifdef CUDA_BACKEND
	
	void updateGLMaterial(const std::shared_ptr<Material>& source, const std::shared_ptr<GLMaterial>& glMaterial)
	{
		glMaterial->baseColor = Vec3f(source->baseColor.r, source->baseColor.g, source->baseColor.b);
		glMaterial->roughness = source->roughness;
		glMaterial->metallic = source->metallic;
		glMaterial->bumpScale = source->bumpScale;
		glMaterial->alpha = source->alpha;
		glMaterial->texColor.load(source->texColor);
		glMaterial->texBump.load(source->texBump);
		glMaterial->texORM.load(source->texORM);
		glMaterial->texAlpha.load(source->texAlpha);
		glMaterial->texEmissiveColor.load(source->texEmissive);
		glMaterial->emissiveIntensity = source->emissiveIntensity;
	}

	void GLTextureMesh::load(const std::shared_ptr<TextureMesh> mesh)
	{
		if (mesh == nullptr)
			return;
		loadMesh(mesh, mVertices, mNormal, mTexCoord, mShapes,0);
		loadMesh(mesh, mVertices_LOD1, mNormal_LOD1, mTexCoord_LOD1, mShapes_LOD1, 1);
		loadMesh(mesh, mVertices_LOD2, mNormal_LOD2, mTexCoord_LOD2, mShapes_LOD2, 2);
	}

	void GLTextureMesh::loadMesh(
		const std::shared_ptr<TextureMesh> mesh, 
		XBuffer<Vec3f>& vertices, 
		XBuffer<Vec3f>& normal, 
		XBuffer<Vec2f>& texCoord,
		std::vector<std::shared_ptr<GLShape>>& shapes,
		int level
	)
	{
		std::shared_ptr<Geometry>& currentGeometry = mesh->lodGeometry(level);
		std::vector<std::shared_ptr<Shape>>& currentShapes = mesh->lodShapes(level);

		vertices.load(currentGeometry->vertices());
		normal.load(currentGeometry->normals());
		texCoord.load(currentGeometry->texCoords());

		uint shapeNum = currentShapes.size();

		if (shapes.size() != shapeNum)
		{
			shapes.resize(shapeNum);
			for (uint i = 0; i < shapeNum; i++)
			{
				shapes[i] = std::make_shared<GLShape>();
			}
		}

		for (uint i = 0; i < shapeNum; i++)
		{
			shapes[i]->glVertexIndex.load(currentShapes[i]->vertexIndex);
			shapes[i]->glNormalIndex.load(currentShapes[i]->normalIndex);
			shapes[i]->glTexCoordIndex.load(currentShapes[i]->texCoordIndex);

			Vec3f S = currentShapes[i]->boundingTransform.scale();
			Mat3f R = currentShapes[i]->boundingTransform.rotation();
			Vec3f T = currentShapes[i]->boundingTransform.translation();

			Mat3f RS = R * Mat3f(
				S[0], 0, 0,
				0, S[1], 0,
				0, 0, S[2]);

			glm::mat4 tm = glm::mat4{
				RS(0, 0), RS(1, 0), RS(2, 0), 0,
				RS(0, 1), RS(1, 1), RS(2, 1), 0,
				RS(0, 2), RS(1, 2), RS(2, 2), 0,
				T[0],	  T[1],	    T[2],	  1 };

			shapes[i]->transform = tm;

			//Setup the material for each shape
			if (currentShapes[i]->material != NULL)
			{
				std::shared_ptr<GLMaterial> currentShapeMtl;
				auto carMtl = std::dynamic_pointer_cast<CarMaterial> (currentShapes[i]->material);

				if (carMtl)
				{
					auto glCarMaterial = std::make_shared<GLCarMaterial>();
					currentShapeMtl = glCarMaterial;
					glCarMaterial->texLightMask.load(carMtl->texLightMask);

				}
				else
				{
					currentShapeMtl = std::make_shared<GLMaterial>();
				}

				updateGLMaterial(currentShapes[i]->material, currentShapeMtl);

				shapes[i]->material = currentShapeMtl;
			}
			else
			{
				shapes[i]->material = NULL;
			}
		}
	}
#endif



	void GLTextureMesh::updateGL()
	{
		if (!mInitialized)
			create();

		mVertices.updateGL();
		mNormal.updateGL();
		mTexCoord.updateGL();

		mVertices_LOD1.updateGL();
		mNormal_LOD1.updateGL();
		mTexCoord_LOD1.updateGL();

		mVertices_LOD2.updateGL();
		mNormal_LOD2.updateGL();
		mTexCoord_LOD2.updateGL();

		for (uint i = 0; i < mShapes.size(); i++)
		{
			mShapes[i]->updateGL();
		}

		for (uint i = 0; i < mShapes_LOD1.size(); i++)
		{
			mShapes_LOD1[i]->updateGL();
		}

		for (uint i = 0; i < mShapes_LOD2.size(); i++)
		{
			mShapes_LOD2[i]->updateGL();
		}
	}

	XBuffer<Vec3f>& GLTextureMesh::verticesLOD(int level)
	{
		switch (level)
		{
		case 0:
			return mVertices;
			break;
		case 1:
			return mVertices_LOD1;
			break;
		case 2:
			return mVertices_LOD2;
			break;

		default:
			break;
		}
	}

	XBuffer<Vec3f>& GLTextureMesh::normalsLOD(int level)
	{
		switch (level)
		{
		case 0:
			return mNormal;
			break;
		case 1:
			return mNormal_LOD1;
			break;
		case 2:
			return mNormal_LOD2;
			break;
		default:
			break;
		}
	}

	XBuffer<Vec2f>& GLTextureMesh::texCoordsLOD(int level)
	{
		switch (level)
		{
		case 0:
			return mTexCoord;
			break;
		case 1:
			return mTexCoord_LOD1;
			break;
		case 2:
			return mTexCoord_LOD2;
			break;
		default:
			break;
		}
	}

	std::vector<std::shared_ptr<GLShape>>& GLTextureMesh::shapesLOD(int level)
	{
		switch (level)
		{
		case 0:
			return mShapes;
			break;
		case 1:
			return mShapes_LOD1;
			break;
		case 2:
			return mShapes_LOD2;
			break;
		default:
			break;
		}
	}

}