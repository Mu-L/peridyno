#pragma once

#include <Node/Agent.h>
#include <Camera.h>

namespace dyno
{
	class ImCameraAgent : public Agent
	{
		DECLARE_CLASS(ImCameraAgent);

	public:
		ImCameraAgent();
		~ImCameraAgent() override;

		DEF_NODE_PORT(Agent, Parent, "");

		std::shared_ptr<Camera> camera() { return mCamera; }

	public:
		DEF_VAR(bool, Enabled, true, "Enable the camera agent");

		DEF_VAR(Vec2i, ViewportLocation, Vec2i(0), "Bottom left corner of the viewport");

	private:
		std::shared_ptr<Camera>		mCamera;
	};
}