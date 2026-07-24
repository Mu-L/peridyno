#include "ImCameraAgent.h"

#include "TrackballCamera.h"

namespace dyno
{
	IMPLEMENT_CLASS(ImCameraAgent);

	ImCameraAgent::ImCameraAgent()
		: Agent()
	{
		mCamera = std::make_shared<TrackballCamera>();
		mCamera->setWidth(128);
		mCamera->setHeight(128);
		mCamera->registerPoint(0, 0);
		mCamera->rotateToPoint(-32, 12);

		//TODO: does not work yet, need to check the validation of the callback function
		this->varViewportSize()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				auto vSize = this->varViewportSize()->getValue();
				mCamera->setWidth(vSize.x);
				mCamera->setHeight(vSize.y);
			}));

		this->varViewportSize()->setValue(Vec2i(256, 256));
	}

	ImCameraAgent::~ImCameraAgent()
	{

	}

	void ImCameraAgent::resetStates()
	{
		updateCamera();
	}

	void ImCameraAgent::updateStates()
	{
		updateCamera();
	}

	void ImCameraAgent::updateCamera()
	{
		auto parent = this->importParent()->getDerivedNode();
		if (parent != nullptr)
		{
			Vec3f loc = this->varLocalTranslation()->getValue();
			Vec3f rot = this->varLocalRotation()->getValue();

			Quat1f quat = Quat1f::fromEulerAngles(rot.x / 180.0f * M_PI, rot.y / 180.0f * M_PI, rot.z / 180.0f * M_PI);

			Vec3f parentLoc = parent->stateLocation()->getValue();
			Quat1f parentRot = parent->stateRotation()->getValue();

			Vec3f worldLoc = parentLoc + parentRot.rotate(loc);
			Quat1f worldRot = parentRot * quat;
			worldRot.normalize();

			this->stateLocation()->setValue(worldLoc);
			this->stateRotation()->setValue(worldRot);

			mCamera->setEyePos(worldLoc);
			mCamera->setTargetPos(worldLoc + worldRot.rotate(Vec3f(0, 0, 1)));
		}
	}
}