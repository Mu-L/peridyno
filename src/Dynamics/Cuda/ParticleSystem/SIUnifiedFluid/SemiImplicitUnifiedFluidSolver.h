/**
 * Copyright 2025 Shusen Liu @ISCAS
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
#include "../Module/ParticleApproximation.h"
#include "ParticleSystem/Module/Kernel.h"
#include "Collision/Attribute.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class SemiImplicitUnifiedFluidSolver : public ParticleApproximation<TDataType>
	{

		DECLARE_TCLASS(SemiImplicitUnifiedFluidSolver, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		SemiImplicitUnifiedFluidSolver();
		~SemiImplicitUnifiedFluidSolver() override;

	public:
		DEF_VAR(Real, Lambda, Real(1), "A variable controlling the normal viscosity");

		DEF_VAR(Real, Mu, Real(0), "A variable controlling the shear viscosity");

		DEF_VAR(Real, Kappa, Real(1), "A variable controlling the strength of const density constraint");

		DEF_VAR(Real, Gamma, Real(10), "A variable controlling the the strength of surface tension");

		DEF_VAR(int, MaxIterationNumber, 100, "");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_VAR_IN(Real, TimeStep, "Time step size!");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
		
		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attributes");

		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Final particle density");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		DEF_VAR(bool, ViscosityDisable, false, "");

		DEF_VAR(bool, DensityConstraintDisable, false, "");

		DEF_VAR(bool, LineSearchDisable, false, "");

		DEF_VAR(bool, SurfaceTensionDisable, false, "");

		DEF_VAR(Real, SolidAdhension, 0.0f, "");

		DEF_VAR(Coord, Gravity, Coord(0.0f, -9.8f, 0.0f), "");

		DECLARE_ENUM(
		EPairwiseFunc,
			Adsorption = 0,
			Bidirection = 1
			);

		DEF_ENUM(EPairwiseFunc, PairwiseFunc, 1, "");

	public:

		void compute() override;

	private:

        void resizeArrays();

		std::shared_ptr<SummationDensity<TDataType>> m_summation;

		DArray<Coord> m_Source;
		DArray<Matrix> m_matL;
		DArray<Matrix> m_matK;
		DArray<Coord> mX;
		DArray<Coord> mY;
		DArray<Coord> mYStar;
	};

	IMPLEMENT_TCLASS(SemiImplicitUnifiedFluidSolver, TDataType)
}