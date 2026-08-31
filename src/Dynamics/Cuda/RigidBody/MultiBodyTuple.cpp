#include "MultiBodyTuple.h"

namespace dyno
{
	void ShapeTuple::updateRange()
	{
		this->varDensity()->setRange(0, 1000);
		this->varRadius()->setRange(0, 100);
		this->varCapsuleLength()->setRange(0, 100);
	}

	void RigidBodyTuple::updateRange()
	{
		this->varFriction()->setRange(0, 1000);
		this->varRestitution()->setRange(FLT_MIN, FLT_MAX);
	}
}
