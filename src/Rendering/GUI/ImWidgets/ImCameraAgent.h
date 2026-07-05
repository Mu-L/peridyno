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

		DEF_VAR(Vec2i, ViewportLocation, Vec2i(0, 512), "Bottom left corner of the viewport");
		DEF_VAR(Vec2i, ViewportSize, Vec2i(256, 256), "Viewport size");

		DEF_VAR(Vec3f, LocalTranslation, Vec3f(0), "");
		DEF_VAR(Vec3f, LocalRotation, Vec3f(0), "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		void updateCamera();

		std::shared_ptr<Camera>		mCamera;
	};
}