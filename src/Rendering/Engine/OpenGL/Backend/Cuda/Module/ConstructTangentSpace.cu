#include "ConstructTangentSpace.h"

namespace dyno {

	IMPLEMENT_CLASS(ConstructTangentSpace)

	ConstructTangentSpace::ConstructTangentSpace()
	{
		this->outNormal()->allocate();
		this->outTangent()->allocate();
		this->outBitangent()->allocate();
		this->outNormalLOD1 ()->allocate();
		this->outTangentLOD1()->allocate();
		this->outBitangentLOD1()->allocate();
		this->outNormalLOD2()->allocate();
		this->outTangentLOD2()->allocate();
		this->outBitangentLOD2()->allocate();
	}

	ConstructTangentSpace::~ConstructTangentSpace()
	{
	}

	template<typename TriangleIndex>
	__global__ void CTS_AccumulateTangentSpace(
		DArray<Vec3f> tangent,
		DArray<Vec3f> bitangent,
		DArray<Vec3f> vertex,
		DArray<Vec3f> normal,
		DArray<Vec2f> texCoord,
		DArray<TriangleIndex> vIndex,
		DArray<TriangleIndex> nIndex,
		DArray<TriangleIndex> tIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vIndex.size()) return;

		TriangleIndex vIdx = vIndex[tId];
		TriangleIndex nIdx = nIndex[tId];
		TriangleIndex tIdx = tIndex[tId];

		Vec3f v0 = vertex[vIdx[0]];
		Vec3f v1 = vertex[vIdx[1]];
		Vec3f v2 = vertex[vIdx[2]];

		Vec2f uv0 = texCoord[tIdx[0]];
		Vec2f uv1 = texCoord[tIdx[1]];
		Vec2f uv2 = texCoord[tIdx[2]];

		// Edges of the triangle : position delta
		Vec3f deltaPos1 = v1 - v0;
		Vec3f deltaPos2 = v2 - v0;

		// UV delta
		Vec2f deltaUV1 = uv1 - uv0;
		Vec2f deltaUV2 = uv2 - uv0;

		float r = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
		Vec3f tangent_i = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
		Vec3f bitangent_i = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * r;

		//Accumulate tangent
		atomicAdd(&tangent[nIdx[0]].x, tangent_i.x);
		atomicAdd(&tangent[nIdx[0]].y, tangent_i.y);
		atomicAdd(&tangent[nIdx[0]].z, tangent_i.z);

		atomicAdd(&tangent[nIdx[1]].x, tangent_i.x);
		atomicAdd(&tangent[nIdx[1]].y, tangent_i.y);
		atomicAdd(&tangent[nIdx[1]].z, tangent_i.z);

		atomicAdd(&tangent[nIdx[2]].x, tangent_i.x);
		atomicAdd(&tangent[nIdx[2]].y, tangent_i.y);
		atomicAdd(&tangent[nIdx[2]].z, tangent_i.z);

		//Accumulate bitangent
		atomicAdd(&bitangent[nIdx[0]].x, bitangent_i.x);
		atomicAdd(&bitangent[nIdx[0]].y, bitangent_i.y);
		atomicAdd(&bitangent[nIdx[0]].z, bitangent_i.z);

		atomicAdd(&bitangent[nIdx[1]].x, bitangent_i.x);
		atomicAdd(&bitangent[nIdx[1]].y, bitangent_i.y);
		atomicAdd(&bitangent[nIdx[1]].z, bitangent_i.z);

		atomicAdd(&bitangent[nIdx[2]].x, bitangent_i.x);
		atomicAdd(&bitangent[nIdx[2]].y, bitangent_i.y);
		atomicAdd(&bitangent[nIdx[2]].z, bitangent_i.z);
	}

	__global__ void CTS_NormalizeTangentSpace(
		DArray<Vec3f> tangent,
		DArray<Vec3f> bitangent,
		DArray<Vec3f> normals)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tangent.size()) return;

		Vec3f normal_i = normals[tId];
		Vec3f tangent_i = tangent[tId];
		Vec3f bitangent_i = bitangent[tId];

		tangent_i = tangent_i - tangent_i.dot(normal_i) * normal_i;
		bitangent_i = bitangent_i - bitangent_i.dot(normal_i) * normal_i;

		tangent[tId] = tangent_i.normalize();
		bitangent[tId] = bitangent_i.normalize();
	}

	void ConstructTangentSpace::compute()
	{
		computeTangent(0);
		computeTangent(1);
		computeTangent(2);
	}

	void ConstructTangentSpace::computeTangent(int level)
	{
		auto mesh = this->inTextureMesh()->constDataPtr();

		if (mesh->lodGeometry(level)->vertices().isEmpty())
			return;

		auto& inVertex = mesh->lodGeometry(level)->vertices();
		auto& inNormal = mesh->lodGeometry(level)->normals();
		auto& inTexCoord = mesh->lodGeometry(level)->texCoords();
		auto& inShapes = mesh->shapes();

		FArray<Vec3f, GPU>* NormalPtr = NULL;
		FArray<Vec3f, GPU>* TangentPtr = NULL;
		FArray<Vec3f, GPU>* BitangentPtr = NULL;
		switch (level)
		{
		case 0:
			NormalPtr = this->outNormal();
			TangentPtr = this->outTangent();
			BitangentPtr = this->outBitangent();
			break;
		case 1:
			NormalPtr = this->outNormalLOD1();
			TangentPtr = this->outTangentLOD1();
			BitangentPtr = this->outBitangentLOD1();
			break;
		case 2:
			NormalPtr = this->outNormalLOD2();
			TangentPtr = this->outTangentLOD2();
			BitangentPtr = this->outBitangentLOD2();
			break;
		default:
			break;
		}

		if (NormalPtr->size() != inNormal.size()) {
			auto totalNum = inNormal.size();
			NormalPtr->resize(totalNum);
			TangentPtr->resize(totalNum);
			BitangentPtr->resize(totalNum);
		}

		auto& outNormal = NormalPtr->getData();
		auto& outTangent = TangentPtr->getData();
		auto& outBitangent = BitangentPtr->getData();

		outNormal.reset();
		outTangent.reset();
		outBitangent.reset();

		for (auto i = 0; i < inShapes.size(); i++)
		{
			auto& shape = inShapes[i];

			auto& vIndex = shape->vertexIndex;
			auto& nIndex = shape->normalIndex;
			auto& tIndex = shape->texCoordIndex;

			cuExecute(nIndex.size(),
				CTS_AccumulateTangentSpace,
				outTangent,
				outBitangent,
				inVertex,
				inNormal,
				inTexCoord,
				vIndex,
				nIndex,
				tIndex);
		}

		cuExecute(outTangent.size(),
			CTS_NormalizeTangentSpace,
			outTangent,
			outBitangent,
			inNormal);

		outNormal.assign(inNormal);
	}

}