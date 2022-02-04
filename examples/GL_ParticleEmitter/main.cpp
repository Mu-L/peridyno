#include <GlfwApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/ParticleEmitterSquare.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	//Create a particle emitter
	auto emitter = std::make_shared<ParticleEmitterSquare<DataType3f>>();
	emitter->varLocation()->setValue(Vec3f(0.5f));

	//Create a particle-based fluid solver
	auto fluid = std::make_shared<ParticleFluid<DataType3f>>();
	fluid->loadParticles(Vec3f(0.0f), Vec3f(0.2f), 0.005f);
	fluid->addParticleEmitter(emitter);

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Vec3f(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	ptRender->setColorMapRange(0, 5);

	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->currentTopology()->connect(ptRender->inPointSet());
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	colorMapper->outColor()->connect(ptRender->inColor());

	fluid->graphicsPipeline()->pushModule(calculateNorm);
	fluid->graphicsPipeline()->pushModule(colorMapper);
	fluid->graphicsPipeline()->pushModule(ptRender);

	//Create a container
	auto container = scene.createNewScene<StaticBoundary<DataType3f>>();
	container->loadCube(Vec3f(0.0f), Vec3f(1.0), 0.02, true);
	container->addParticleSystem(fluid);
}

int main()
{
	CreateScene();

	GLRenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1280, 768);
	window.mainLoop();

	delete engine;

	return 0;
}


