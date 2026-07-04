/**
 * Copyright 2026 Xiaowei He
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
#include "Node.h"

#include "Quat.h"

namespace dyno
{
	/**
	 * @brief Agent is a special kind of node that has a location and rotation in the world coordinate system.
	 *		  It serves as the base class for all intelligent agents in the physical environments, such as sensors, policies, and actuators.
	 */

	class Agent : virtual public Node
	{
	public:
		Agent();

		std::string getNodeType() override { return "Agents"; }

	public:
		DEF_VAR_STATE(Vec3f, Location, Vec3f(0), "Node location");
		DEF_VAR_STATE(Quat1f, Rotation, Quat1f(), "Node rotation");
	};
}