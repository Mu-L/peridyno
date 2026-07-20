#include <GlfwApp.h>
#include <SceneGraph.h>

//Render
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include <Module/CalculateNorm.h>

//Shape
#include <BasicShapes/CubeModel.h>

//Point Sampler
#include <Samplers/ShapeSampler.h>
#include <Commands/Extrude.h>

//Collision and Boundary
#include <Collision/Attribute.h>
#include <Collision/NeighborPointQuery.h>
#include <Multiphysics/VolumeBoundary.h>
#include <Volume/BasicShapeToVolume.h>

//ParticleSystem
#include <ParticleSystem/Module/ParticleIntegrator.h>
#include <ParticleSystem/SIUnifiedFluid/SemiImplicitUnifiedFluidSolver.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/ParticleFluid.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(4.5, 4, 4.5));
	scn->setLowerBound(Vec3f(-4.5, -4, -4.5));
	scn->setGravity(Vec3f(0.0f));

	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.06, 0.0));
	cube->varLength()->setValue(Vec3f(0.08, 0.08, 0.08));
	cube->graphicsPipeline()->disable();

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->setVisible(false);

	cube->connect(sampler->importShape());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->varReshuffleParticles()->setValue(true);
	fluid->varSmoothingLength()->setValue(3.0);
	initialParticles->connect(fluid->importInitialStates());

	{
		fluid->animationPipeline()->clear();
		fluid->varSamplingDistance()->setValue(0.005f);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		nbrQuery->varSizeLimit()->setValue(0);
		fluid->stateSmoothingLength()->connect(nbrQuery->inRadius());
		fluid->statePosition()->connect(nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(nbrQuery);

		auto uniSolver = std::make_shared<SemiImplicitUnifiedFluidSolver<DataType3f>>();
		uniSolver->varMaxIterationNumber()->setValue(20);
		uniSolver->varKernelType()->getDataPtr()->setCurrentKey(1);
		uniSolver->varLambda()->setValue(0.20f);
		uniSolver->varMu()->setValue(0.0);
		uniSolver->varKappa()->setValue(1.0f);
		uniSolver->varGamma()->setValue(2000.0f);
		uniSolver->varSurfaceTensionDisable()->setValue(false);
		uniSolver->varViscosityDisable()->setValue(false);
		uniSolver->varDensityConstraintDisable()->setValue(false);
		uniSolver->varLineSearchDisable()->setValue(true);
		fluid->stateTimeStep()->connect(uniSolver->inTimeStep());
		fluid->stateSmoothingLength()->connect(uniSolver->inSmoothingLength());
		fluid->stateSamplingDistance()->connect(uniSolver->inSamplingDistance());
		fluid->statePosition()->connect(uniSolver->inPosition());
		fluid->stateVelocity()->connect(uniSolver->inVelocity());
		nbrQuery->outNeighborIds()->connect(uniSolver->inNeighborIds());
		fluid->animationPipeline()->pushModule(uniSolver);

	}

	auto volBoundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	fluid->connect(volBoundary->importParticleSystems());

	//Create a cube for boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 1.0f, 0.0f));
	cubeBoundary->varLength()->setValue(Vec3f(1.2f, 2.0f, 1.2f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());
	cube2vol->connect(volBoundary->importVolumes());

	fluid->graphicsPipeline()->clear();

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->varBaseColor()->setValue(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	ptRender->varPointSize()->setValue(0.001);
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
	app.initialize(1920, 1080);

	app.renderWindow()->getCamera()->setEyePos(Vec3f(1.09f, 0.53f, 0.60f));
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(0.01f, 0.02f, -0.06f));

	app.mainLoop();

	return 0;
}