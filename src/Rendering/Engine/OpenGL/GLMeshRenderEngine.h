#pragma once

#include <memory>

#include "GLRenderEngine.h"

#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"
#include "GLPhotorealisticRender.h"
#include "GLSurfaceVisualModule.h"
#include "SceneGraph.h"

namespace dyno
{
	class GLMeshRenderEngine : public GLRenderEngine
	{
	public:
		GLMeshRenderEngine();
		~GLMeshRenderEngine();

		void addField(FBase* field);

		void addNode(std::shared_ptr<Node> node);

		virtual std::string name() const override;

		std::shared_ptr<SceneGraph> renderSceneGraph = NULL;

		void resetScene() { renderSceneGraph->reset(); }
	};
};
