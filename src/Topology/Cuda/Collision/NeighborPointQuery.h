/**
 * Copyright 2021 Xiaowei He
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

#include "Primitive/Primitive3D.h"

namespace dyno 
{
	template<typename TDataType>
	class NeighborPointQuery : public ComputeModule
	{
		DECLARE_TCLASS(NeighborPointQuery, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;

		NeighborPointQuery();
		~NeighborPointQuery() override;

	public:
		DECLARE_ENUM(Spatial,
			UNIFORM = 0,
			BVH = 1,
			OCTREE = 2);

		DEF_ENUM(Spatial, Spatial, Spatial::UNIFORM, "");

		DEF_VAR(uint, SizeLimit, 0, "Maximum number of neighbors");

		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_VAR_IN(Real, Radius, "Search radius");

		/**
		 * @brief A set of points whose neighbors will be required for.
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "A set of points whose neighbors will be required for");

		/**
		 * @brief Another set of points the algorithm will require neighbor ids for.
		 *		  If not set, the set of points in Position will be required.
		 */
		DEF_ARRAY_IN(Coord, Other, DeviceType::GPU, 
			"Another set of points the algorithm will require neighbor ids for. If not set, the set of points in Position will be required.");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");

	protected:
		void compute() override;

	private:
		void requestDynamicNeighborIds();

		void requestFixedSizeNeighborIds();

		void requestNeighborIdsWithBVH();

		void requestNeighborIdsWithOctree();
	};
}