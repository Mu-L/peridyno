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
#include "OceanBase.h"

#include "Field/FilePath.h"

#include "Topology/TriangleSet.h"
#include "Topology/HeightField.h"

namespace dyno
{
	template<typename TDataType>
	class LargeOcean : public OceanBase<TDataType>
	{
		DECLARE_TCLASS(LargeOcean, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename ::dyno::Vector<Real, 2> Coord2D;
		typedef typename ::dyno::Vector<Real, 3> Coord3D;
		typedef typename ::dyno::Vector<Real, 4> Coord4D;

		LargeOcean();
		~LargeOcean() override;

	public:
		DEF_VAR(FilePath, FileName, "", "File name for the ocean mesh");

	public:
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "Topology");

		DEF_ARRAY_STATE(Coord2D, TexCoord, DeviceType::GPU, "");

		DEF_ARRAY2D_STATE(Coord4D, BumpMap, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(HeightField<TDataType>, HeightField, "");

	protected:
		void resetStates() override;

		void updateStates() override;

	private:
		std::string mFileName;
	};

	IMPLEMENT_TCLASS(LargeOcean, TDataType)
}