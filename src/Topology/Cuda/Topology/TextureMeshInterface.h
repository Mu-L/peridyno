/**
 * Copyright 2022 Shusen Liu
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
 *
 * Revision history:
 *
 * 2024-02-03: replace TriangleSet with PolygonSet as the major state;
 */

#pragma once
#include "Node/ParametricModel.h"

#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"
#include "Topology/TextureMesh.h"

namespace dyno
{

	class TextureMeshInterface
	{
	public:

		TextureMeshInterface() {};
		~TextureMeshInterface() {};

		virtual FInstance<TextureMesh>* getTextureMesh() = 0;

	};

	
}