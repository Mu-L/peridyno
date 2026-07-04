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
	}

	ImCameraAgent::~ImCameraAgent()
	{

	}
}