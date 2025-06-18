
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector.h"

#include "OBase.h"
#include "Module.h"
#include "Topology/TextureMesh.h"

#include "Field/Canvas.h"

namespace dyno {

	class SkinInfo : public OBase
	{

	public:

		SkinInfo() {};

		~SkinInfo() 
		{
			this->clear();
		};

		void pushBack_Data(const std::vector<Vec4f>& Weight_0,
			const std::vector<Vec4f>& Weight_1,
			const std::vector<Vec4f>& ID_0,
			const std::vector<Vec4f>& ID_1,
			std::vector<Vec4f>* Weight_2 = nullptr,
			std::vector<Vec4f>* ID_2 = nullptr
		) 
		{
			skinNum++;

			V_jointWeight_0.resize(skinNum);
			V_jointWeight_1.resize(skinNum);
			V_jointID_0.resize(skinNum);
			V_jointID_1.resize(skinNum);

			this->V_jointWeight_0[skinNum-1].assign(Weight_0);
			this->V_jointWeight_1[skinNum - 1].assign(Weight_1);
			this->V_jointID_0[skinNum - 1].assign(ID_0);
			this->V_jointID_1[skinNum - 1].assign(ID_1);

			if (Weight_2 && ID_2) 
			{
				V_jointWeight_2.resize(skinNum);
				V_jointID_2.resize(skinNum);
				this->V_jointWeight_2[skinNum - 1].assign(*Weight_2);
				this->V_jointID_2[skinNum - 1].assign(*ID_2);				
			}
		}

		void clear() 
		{
			for (size_t i = 0; i < skinNum; i++)
			{
				V_jointWeight_0[i].clear();
				V_jointWeight_1[i].clear();
				if(!V_jointWeight_2.empty())
					V_jointWeight_2[i].clear();

				V_jointID_0[i].clear();
				V_jointID_1[i].clear();
				if (!V_jointID_2.empty())
					V_jointID_2[i].clear();

			}

			skinNum = 0;

			V_jointWeight_0.clear();
			V_jointWeight_1.clear();
			V_jointWeight_2.clear();

			V_jointID_0.clear();
			V_jointID_1.clear();
			V_jointID_2.clear();


			skin_VerticeRange.clear();
		}

		int size() { return skinNum; };

		bool isEmpty()
		{
			if (skinNum == 0 || mesh == NULL || initialPosition.isEmpty() || initialNormal.isEmpty())
				return true;
		}

		std::vector<DArray<Vec4f>> V_jointWeight_0;
		std::vector<DArray<Vec4f>> V_jointWeight_1;
		std::vector<DArray<Vec4f>> V_jointWeight_2;

		std::vector<DArray<Vec4f>> V_jointID_0;
		std::vector<DArray<Vec4f>> V_jointID_1;
		std::vector<DArray<Vec4f>> V_jointID_2;

		std::map<int, Vec2u> skin_VerticeRange;

		std::shared_ptr<TextureMesh> mesh = NULL;

		DArray<Vec3f> initialPosition;

		DArray<Vec3f> initialNormal;


	private:

		int skinNum = 0;

		
	};


}

