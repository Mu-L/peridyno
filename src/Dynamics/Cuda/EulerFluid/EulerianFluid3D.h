#pragma once
#include "Node.h"

#include "Topology/PhaseField.h"

#include "PhaseField/PFKernel.h"

namespace dyno
{
	template<typename TDataType>
	class EulerianFluid3D : public Node
	{
		DECLARE_TCLASS(EulerianFluid3D, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		EulerianFluid3D();
		~EulerianFluid3D() override;

		DEF_VAR(Vec3i, Dimension, Vec3i(64), "");

	public:
		DEF_ARRAY3D_STATE(Real, Mass, DeviceType::GPU, "");

		DEF_ARRAY3D_STATE(Coord, Velocity, DeviceType::GPU, "");
		
		DEF_INSTANCE_STATE(PhaseField<TDataType>, PhaseField, "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		DArray3D<Real> vel_u;
		DArray3D<Real> vel_v;
		DArray3D<Real> vel_w;

		DArray3D<Real> pre_vel_u;
		DArray3D<Real> pre_vel_v;
		DArray3D<Real> pre_vel_w;

		DArray3D<Coord> velBuf;
		DArray3D<Coord> velSrc;

		DArray3D<Real> omega;

		DArray3D<Coord> dir;

		DArray3D<Coef> mat;
		DArray3D<Real> RHS;
		DArray3D<Real> pressure;

		//mass buffer
		DArray3D<Real> mb0;
		DArray3D<Real> mb1;

		Grid4f pigments;
	};
}