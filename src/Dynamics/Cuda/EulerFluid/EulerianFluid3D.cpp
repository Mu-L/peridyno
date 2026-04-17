#include "EulerianFluid3D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(EulerianFluid3D, TDataType)

	template<typename TDataType>
	EulerianFluid3D<TDataType>::EulerianFluid3D()
		: Node()
	{
		this->statePhaseField()->allocate();
	}

	template<typename TDataType>
	EulerianFluid3D<TDataType>::~EulerianFluid3D()
	{
	}

	template<typename TDataType>
	void EulerianFluid3D<TDataType>::resetStates()
	{
		auto pf = this->statePhaseField()->getDataPtr();

		auto dim = this->varDimension()->getValue();

		this->stateVelocity()->resize(dim.x, dim.y, dim.z);
		this->stateVelocity()->reset();

		this->stateMass()->resize(dim.x, dim.y, dim.z);
		this->stateMass()->reset();

		omega.resize(dim.x, dim.y, dim.z);

		vel_u.resize(dim.x + 1, dim.y, dim.z);
		vel_v.resize(dim.x, dim.y + 1, dim.z);
		vel_w.resize(dim.x, dim.y, dim.z + 1);

		pre_vel_u.resize(dim.x + 1, dim.y, dim.z);
		pre_vel_v.resize(dim.x, dim.y + 1, dim.z);
		pre_vel_w.resize(dim.x, dim.y, dim.z + 1);

		mat.resize(dim.x, dim.y, dim.z);
		RHS.resize(dim.x, dim.y, dim.z);
		pressure.resize(dim.x, dim.y, dim.z);
		pressure.reset();

		dir.resize(dim.x, dim.y, dim.z);

		mb0.resize(dim.x, dim.y, dim.z);
		mb1.resize(dim.x, dim.y, dim.z);

		velBuf.resize(dim.x, dim.y, dim.z);
		velSrc.resize(dim.x, dim.y, dim.z);

		vel_u.reset();
		vel_v.reset();
		vel_w.reset();

		pf->initialize(dim.x, dim.y, dim.z);

		CArray3D<Real> fraction(dim.x, dim.y, dim.z);
		CArray3D<Coord> initial_vel(dim.x, dim.y, dim.z);
		CArray3D<Real> host_vel(dim.x, dim.y + 1, dim.z);
		CArray3D<Real> host_omega(dim.x, dim.y, dim.z);

		for(int i = 0; i < dim.x; i++)
			for (int j = 0; j <dim.y; j++)
			{
				for (int k = 0; k < dim.z; k++)
				{
					fraction(i, j, k) = (i < 32 && j < 32) ? 1.0f : 0.0f;
					initial_vel(i, j, k) = Coord(0, 0, 0);
					host_vel(i, j, k) = 0;
					host_omega(i, j, k) = 1;
				}
			}

		pf->volumeFraction().assign(fraction);
		this->stateMass()->assign(fraction);
		this->stateVelocity()->assign(initial_vel);
		vel_v.assign(host_vel);
		omega.assign(host_omega);

		fraction.clear();
	}

	template<typename TDataType>
	void EulerianFluid3D<TDataType>::updateStates()
	{
		Real gamma = 0.05;

		auto phase = this->statePhaseField()->getDataPtr();
		auto dt = this->stateTimeStep()->getValue();
		auto h = phase->gridSpacing();
		auto dim = this->varDimension()->getValue();

		auto& mass = this->stateMass()->getData();
		auto& vel = this->stateVelocity()->getData();

		PFKernel::ApplyGravity(vel_u, vel_v, vel_w, Coord(0, -2.0, 0.0), dim.x, dim.y, dim.z, dt);

		PFKernel::SetU(vel_u);
		PFKernel::SetV(vel_v);
		PFKernel::SetW(vel_w);

		PFKernel::PrepareForProjection(vel_u, vel_v, vel_w, mat, RHS, mass, h, dt);
		for (int i = 0; i < 20; i++)
		{
			PFKernel::Projection(pressure, mb0, mat, RHS, 1);
		}
		PFKernel::UpdateVelocity(vel_u, vel_v, vel_w, pressure, mass, h, dt);

		PFKernel::InterpolateVelocity(vel, vel_u, vel_v, vel_w);

		//Advect
		mb0.assign(mass);
		PFKernel::AdvectForward(mass, mb0, vel, dt);

		velBuf.assign(vel);
		velSrc.assign(vel);
		PFKernel::AdvectBackward(vel, velBuf, velSrc, dt);
		PFKernel::InterpolateVelocity(vel_u, vel_v, vel_w, vel);

		//printf("total mass: %f", PFKernel::calcualteTotalMass(mass));

		//Sharpening
		mb0.assign(mass);
		PFKernel::Sharpening(mass, dir, mb0, vel_u, vel_v, vel_w, omega, gamma, h, dt);

		mb0.assign(mass);
		float a = 1.0f * gamma / h / h * dt;// dt;
		PFKernel::Jacobi(mass, mb0, mb1, vel, a, 1.0f + 6.0f * a, 10);

		phase->volumeFraction().assign(mass);
	}

	DEFINE_CLASS(EulerianFluid3D);
}