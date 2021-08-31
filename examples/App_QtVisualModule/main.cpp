#include "QtGUI/QtApp.h"

#include "ParticleSystem/ParticleFluid.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"

#include "module/PointRender.h"
#include "module/CalculateNorm.h"
#include "module/ColorMapping.h"

#include "../VTK/VtkRenderEngine.h"
#include "../VTK/VtkFluidVisualModule.h"

using namespace std;
using namespace dyno;

bool useVTK = true;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vec3f(1.5, 1, 1.5));
	scene.setLowerBound(Vec3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vec3f(-0.5, 0, -0.5), Vec3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../data/bowl/bowl.sdf", false);

	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
	fluid->loadParticles(Vec3f(0.5, 0.2, 0.4), Vec3f(0.7, 1.5, 0.6), 0.005);
	fluid->setMass(100);
	root->addParticleSystem(fluid);

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);

	if (useVTK)
	{
		auto fRender = std::make_shared<FluidVisualModule>();
		//fRender->setColor(1, 0, 0);
		fluid->currentTopology()->connect(fRender->inPointSet());
		fluid->graphicsPipeline()->pushModule(fRender);
	}
	else
	{
		auto ptRender = std::make_shared<PointRenderer>();
		ptRender->setColor(Vec3f(1, 0, 0));
		ptRender->setColorMapMode(PointRenderer::PER_VERTEX_SHADER);
		ptRender->setColorMapRange(0, 5);
		fluid->currentTopology()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());
		fluid->graphicsPipeline()->pushModule(ptRender);
	}

	fluid->currentVelocity()->connect(calculateNorm->inVec());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());

	fluid->graphicsPipeline()->pushModule(calculateNorm);
	fluid->graphicsPipeline()->pushModule(colorMapper);
}

int main()
{
	CreateScene();

	RenderEngine* engine;
	if (useVTK)
	{
		engine = new VtkRenderEngine;
	}
	else
	{
		engine = new RenderEngine;
	}

	QtApp window;
	window.setRenderEngine(engine);
	window.createWindow(1024, 768);

	window.mainLoop();
	return 0;
}