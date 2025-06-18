#include <GlfwApp.h>
#include <SceneGraph.h>
#include <Log.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include "ParticleSystem/Module/SimpleVelocityConstraint.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "Collision/NeighborPointQuery.h"
#include <ColorMapping.h>
#include <ImColorbar.h>
#include "Auxiliary/DataSource.h"

#include <Samplers/ShapeSampler.h>
#include "PointsLoader.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	auto ptsLoader = scn->addNode(std::make_shared<PointsLoader<DataType3f>>());
	ptsLoader->varFileName()->setValue(getAssetPath() + "fish/FishPoints.obj");
	ptsLoader->varRotation()->setValue(Vec3f(0.0f, 0.0f, 3.1415926f));
	ptsLoader->varLocation()->setValue(Vec3f(0.0f, 0.15f, 0.23f));
	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	ptsLoader->outPointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->varReshuffleParticles()->setValue(true);
	initialParticles->connect(fluid->importInitialStates());

	fluid->animationPipeline()->clear();
	{

		fluid->varSmoothingLength()->setValue(2.5);

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		fluid->stateTimeStep()->connect(integrator->inTimeStep());
		fluid->statePosition()->connect(integrator->inPosition());
		fluid->stateVelocity()->connect(integrator->inVelocity());
		fluid->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		fluid->stateSmoothingLength()->connect(nbrQuery->inRadius());
		fluid->statePosition()->connect(nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(nbrQuery);

		auto simple = std::make_shared <SimpleVelocityConstraint<DataType3f>>();
		simple->varViscosity()->setValue(500);
		simple->varSimpleIterationEnable()->setValue(false);
		fluid->stateTimeStep()->connect(simple->inTimeStep());
		fluid->stateSmoothingLength()->connect(simple->inSmoothingLength());
		fluid->statePosition()->connect(simple->inPosition());
		fluid->stateVelocity()->connect(simple->inVelocity());

		simple->inSamplingDistance()->setValue(Real(0.005));
		nbrQuery->outNeighborIds()->connect(simple->inNeighborIds());
		fluid->animationPipeline()->pushModule(simple);
	}

	//Create a boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.5f, 1.0f, 0.5f));
	cubeBoundary->varLength()->setValue(Vec3f(2.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());
	fluid->connect(container->importParticleSystems());

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

	fluid->statePointSet()->connect(ptRender->inPointSet());
	colorMapper->outColor()->connect(ptRender->inColor());

	fluid->graphicsPipeline()->pushModule(ptRender);

	// A simple color bar widget for node
	auto colorBar = std::make_shared<ImColorbar>();
	colorBar->varMax()->setValue(5.0f);
	colorBar->varFieldName()->setValue("Velocity");
	calculateNorm->outNorm()->connect(colorBar->inScalar());
	// add the widget to app
	fluid->graphicsPipeline()->pushModule(colorBar);

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


