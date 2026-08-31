/**
 * Copyright 2024 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "TriangleSet.h"

#include "Field/Color.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	class Material : public Object
	{
	public:

		Material(){};
		~Material() override 
		{
			texColor.clear();
			texBump.clear();
			texORM.clear();
			texAlpha.clear();
			texEmissive.clear();
		};

		Color baseColor = Color::LightGray();
		float metallic = 0;
		float roughness = 0.5;
		float alpha = 1;
		float bumpScale = 1;
		float emissiveIntensity = 0;

		DArray2D<Vec4f> texColor;
		DArray2D<Vec4f> texBump;
		DArray2D<Vec4f> texORM;
		DArray2D<Vec4f> texAlpha;
		DArray2D<Vec4f> texEmissive;
	};


	class Shape : public Object
	{
	public:
		Shape() {};
		Shape(const Shape& other) 
		{
			this->vertexIndex.assign(other.vertexIndex);
			this->normalIndex.assign(other.normalIndex);
			this->texCoordIndex.assign(other.texCoordIndex);
			this->boundingBox = other.boundingBox;
			this->boundingTransform = other.boundingTransform;
			this->material = other.material;
		}
		~Shape() override { clear(); };
		void clear() 
		{
			vertexIndex.clear();
			normalIndex.clear();
			texCoordIndex.clear();
			material = nullptr;
		}


		void assign(std::shared_ptr<Shape> dataPtr)
		{
			if (dataPtr)
			{
				vertexIndex.assign(dataPtr->vertexIndex);
				normalIndex.assign(dataPtr->normalIndex);
				texCoordIndex.assign(dataPtr->texCoordIndex);

				boundingBox = dataPtr->boundingBox;
				boundingTransform = dataPtr->boundingTransform;

				material = dataPtr->material;
			}
		}

		DArray<Topology::Triangle> vertexIndex;
		DArray<Topology::Triangle> normalIndex;
		DArray<Topology::Triangle> texCoordIndex;

		TAlignedBox3D<Real> boundingBox;
		Transform3f boundingTransform;

		std::shared_ptr<Material> material = nullptr;

	};


	class Geometry : public Object
	{
	public:
		Geometry() {};
		~Geometry() { clear(); };
		Geometry(Geometry&& other) noexcept {
			mVertices.assign(other.vertices());
			mNormals.assign(other.normals());
			mTexCoords.assign(other.texCoords());
			mShapeIds.assign(other.shapeIds());
		}
		Geometry& operator=(Geometry&& other) noexcept {
			mVertices.assign(other.vertices());
			mNormals.assign(other.normals());
			mTexCoords.assign(other.texCoords());
			mShapeIds.assign(other.shapeIds());
		}
		void clear() 
		{
			mVertices.clear();
			mNormals.clear();
			mTexCoords.clear();
			mShapeIds.clear();
		}

		void assign(std::shared_ptr<Geometry> dataPtr)
		{
			if (dataPtr) 
			{
				mVertices.assign(dataPtr->vertices());
				mNormals.assign(dataPtr->normals());
				mTexCoords.assign(dataPtr->texCoords());
				mShapeIds.assign(dataPtr->shapeIds());
			}
		}

		DArray<Vec3f>& vertices() { return mVertices; }
		DArray<Vec3f>& normals() { return mNormals; }
		DArray<Vec2f>& texCoords() { return mTexCoords; }
		DArray<uint>& shapeIds() { return mShapeIds; }

	private:
		DArray<Vec3f> mVertices;
		DArray<Vec3f> mNormals;
		DArray<Vec2f> mTexCoords;
		DArray<uint> mShapeIds;
	};

	class TextureMesh : public Topology
	{
	public:
		TextureMesh();
		~TextureMesh() override;

		std::shared_ptr<Geometry>& geometry();

		std::vector<std::shared_ptr<Shape>>& shapes() { return mShapes; }

		std::shared_ptr<Geometry>& lodGeometry(int level = 1);

		std::vector<std::shared_ptr<Shape>>& lodShapes(int level = 1);

		void merge(const std::shared_ptr<TextureMesh> texMesh01, const std::shared_ptr<TextureMesh> texMesh02);

		void clear();

		void safeConvert2TriangleSet(TriangleSet<DataType3f>& triangleSet);

		void convert2TriangleSet(TriangleSet<DataType3f>& triangleSet);

		std::vector<Vec3f> updateTexMeshBoundingBox(int level = 0);

		const bool useLod();

		template<typename Vec3f>
		void transPoint2Vertices(
			DArray<Vec3f>& pAttribute,
			DArray<Vec3f>& vAttribute,
			DArrayList<int>& contactList
		);
		void setLodDistance(int level, float d) 
		{
			switch (level)
			{
			case 0:
				break;
			case 1:
				mDistanceLod1 = d;
				break;
			case 2:
				mDistanceLod2 = d;
			default:
				break;
			}
		}

		int getLodDistance(int level) 
		{
			switch (level)
			{
			case 0:
				return 0;
				break;
			case 1:
				return mDistanceLod1;
				break;
			case 2:
				return mDistanceLod2;
				break;
			default:
				break;
			}
		}

		void assign(std::shared_ptr<TextureMesh> other)
		{

			auto s = std::shared_ptr<Geometry>(new Geometry());
			mMeshData.reset();
			mMeshData = std::shared_ptr<Geometry>(new Geometry());
			mMeshData->assign(other->geometry());

			mShapes.resize(other->shapes().size());
			for (size_t i = 0; i < mShapes.size(); i++)
			{
				mShapes[i] = std::make_shared<Shape>();
				mShapes[i]->assign(other->shapes()[i]);
			}

			mLod1 = std::make_shared<Geometry>();
			mLod1->assign(other->lodGeometry(1));

			mLod1Shapes.resize(other->lodShapes(1).size());
			for (size_t i = 0; i < mLod1Shapes.size(); i++)
			{
				mLod1Shapes[i] = std::make_shared<Shape>();
				mLod1Shapes[i]->assign(other->lodShapes(1)[i]);
			}
			mDistanceLod1 = other->getLodDistance(1);

			mLod2 = std::make_shared<Geometry>();
			mLod2->assign(other->lodGeometry(2));

			mLod2Shapes.resize(other->lodShapes(2).size());
			for (size_t i = 0; i < mLod2Shapes.size(); i++)
			{
				mLod2Shapes[i] = std::make_shared<Shape>();
				mLod2Shapes[i]->assign(other->lodShapes(2)[i]);
			}
			mDistanceLod1 = other->getLodDistance(2);
		}

	private:
		std::shared_ptr<Geometry> mMeshData = NULL;
		std::vector<std::shared_ptr<Shape>> mShapes;

		std::shared_ptr<Geometry> mLod1 = NULL;
		std::vector<std::shared_ptr<Shape>> mLod1Shapes;
		float mDistanceLod1 = 1;

		std::shared_ptr<Geometry> mLod2 = NULL;
		std::vector<std::shared_ptr<Shape>> mLod2Shapes;
		float mDistanceLod2 = 3;

	};

	
};