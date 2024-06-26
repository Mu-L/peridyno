/**
 * Copyright 2017-2021 Xiaowei HE
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
#include "Module/ComputeModule.h"

namespace dyno 
{
	template<typename TDataType>
	class ColorMapping : public ComputeModule
	{
		DECLARE_TCLASS(ColorMapping, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DECLARE_ENUM(ColorTable,
			Jet = 0,
			Heat = 1);

		ColorMapping() {};
		~ColorMapping() override {};

		void compute() override;

	public:
		DEF_ENUM(ColorTable, Type, ColorTable::Jet, "");

		DEF_VAR(Real, Min, Real(0), "");
		DEF_VAR(Real, Max, Real(1), "");

		DEF_ARRAY_IN(Real, Scalar, DeviceType::GPU, "");
		DEF_ARRAY_OUT(Vec3f, Color, DeviceType::GPU, "");
	};
}
