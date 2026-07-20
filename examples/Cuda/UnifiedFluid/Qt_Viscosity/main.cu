#include <UbiApp.h>
#include <SceneGraph.h>

//Render
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include <Module/CalculateNorm.h>

//Shape
#include <BasicShapes/CubeModel.h>
#include <BasicShapes/PlaneModel.h>

//Point Sampler
#include <Samplers/ShapeSampler.h>
#include <Commands/Extrude.h>

//Collision and Boundary
#include <Collision/Attribute.h>
#include <Collision/NeighborPointQuery.h>
#include <Multiphysics/VolumeBoundary.h>
#include <Volume/BasicShapeToVolume.h>
#include <Volume/BasicShapeToVolume.h>

//ParticleSystem
#include <ParticleSystem/SIUnifiedFluid/SemiImplicitUnifiedFluidSolver.h>
#include <ParticleSystem/PdGhostUnifiedFluid.h>
#include <ParticleSystem/MakeGhostParticles.h>
#include <ParticleSystem/GhostFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>

//Particle Emitter
#include <ParticleSystem/Emitters/CircularEmitter.h>

//Others
#include <Auxiliary/DataSource.h>

using namespace std;
using namespace dyno;

template<typename TDataType>
class UpwardNormalMakeGhostParticles : public MakeGhostParticles<TDataType>
{
public:
	typedef typename TDataType::Coord Coord;

protected:
	void resetStates() override
	{
		auto& inTopo = this->inPoints()->getData();
		int num = inTopo.getPoints().size();

		std::vector<Coord> hostNormal(num, Coord(0, 1, 0));
		this->stateNormal()->resize(num);
		this->stateNormal()->assign(hostNormal);

		MakeGhostParticles<TDataType>::resetStates();
	}
};

std::shared_ptr<SceneGraph> createScene()
{

	Real vis_uni_Lambda = 2000.0f;
	Real vis_uni_Mu = 0.0f;
	Real ST_gamma = 0.0;
	Real Rho_Lambda = 1.00f;

	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(10.5, 10.0, 10.5));
	scn->setLowerBound(Vec3f(-10.5, -10.0, -10.5));
	scn->setGravity(Vec3f(0.0f, -9.0f, 0.0f));

	auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube1->varLocation()->setValue(Vec3f(0.0, -0.01, 0.0));
	cube1->varLength()->setValue(Vec3f(0.5, 0.02, 0.5));
	cube1->graphicsPipeline()->disable();
	auto sampler1 = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	sampler1->varSamplingDistance()->setValue(0.005);
	sampler1->graphicsPipeline()->disable();
	cube1->connect(sampler1->importShape());
	auto GhostPoints1 = scn->addNode(std::make_shared<UpwardNormalMakeGhostParticles<DataType3f>>());
	GhostPoints1->varReverseNormal()->setValue(false);
	sampler1->statePointSet()->promoteOuput()->connect(GhostPoints1->inPoints());
	GhostPoints1->graphicsPipeline()->disable();

	auto Emitter = scn->addNode(std::make_shared<CircularEmitter<DataType3f>>());
	Emitter->varScale()->setValue(Vec3f(0.12f, 0.12f, 0.12f));
	Emitter->varLocation()->setValue(Vec3f(0.0f, 0.279f, 0.0f));
	Emitter->varVelocityMagnitude()->setValue(2.0f);
	Emitter->varSpacing()->setValue(1.0f);

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());

	auto fluid = scn->addNode(std::make_shared<PdGhostUnifiedFluid<DataType3f>>());
	Emitter->connect(fluid->importParticleEmitters());
	//initialParticles->connect(fluid->importInitialStates());
	GhostPoints1->connect(fluid->importBoundaryParticles());

	fluid->animationPipeline()->clear();
	fluid->varSmoothingLength()->setValue(2.5);
	fluid->varSamplingDistance()->setValue(0.005);

	auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
	fluid->stateSmoothingLength()->connect(nbrQuery->inRadius());
	fluid->statePositionMerged()->connect(nbrQuery->inPosition());
	fluid->animationPipeline()->pushModule(nbrQuery);

	auto uniSolver = std::make_shared<SemiImplicitUnifiedFluidSolver<DataType3f>>();
	uniSolver->varKernelType()->getDataPtr()->setCurrentKey(1);
	uniSolver->varMaxIterationNumber()->setValue(30);
	uniSolver->varLambda()->setValue(vis_uni_Lambda);
	uniSolver->varMu()->setValue(vis_uni_Mu);
	uniSolver->varKappa()->setValue(Rho_Lambda);
	uniSolver->varGamma()->setValue(ST_gamma);
	uniSolver->varSurfaceTensionDisable()->setValue(true);
	uniSolver->varSolidAdhension()->setValue(0.0f);
	uniSolver->varViscosityDisable()->setValue(false);
	uniSolver->varDensityConstraintDisable()->setValue(false);
	fluid->stateTimeStep()->connect(uniSolver->inTimeStep());
	fluid->stateSmoothingLength()->connect(uniSolver->inSmoothingLength());
	fluid->stateSamplingDistance()->connect(uniSolver->inSamplingDistance());
	fluid->stateAttributeMerged()->connect(uniSolver->inAttribute());
	fluid->statePositionMerged()->connect(uniSolver->inPosition());
	fluid->stateVelocityMerged()->connect(uniSolver->inVelocity());
	nbrQuery->outNeighborIds()->connect(uniSolver->inNeighborIds());
	fluid->animationPipeline()->pushModule(uniSolver);

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->graphicsPipeline()->clear();
	auto sufaceRender = std::make_shared<GLSurfaceVisualModule>();
	sufaceRender->varBaseColor()->setValue(Color(1.0f));
	sufaceRender->varAlpha()->setValue(0.0f);
	plane->stateTriangleSet()->connect(sufaceRender->inTriangleSet());
	plane->graphicsPipeline()->pushModule(sufaceRender);

	//Create a container
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.0f));
	cubeBoundary->varLength()->setValue(Vec3f(1.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	cube2vol->connect(container->importVolumes());
	fluid->connect(container->importParticleSystems());

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);

	app.setSceneGraph(createScene());
	app.initialize(1920, 1080);

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->setUseEnvmapBackground(false);
		renderer->setEnvmapScale(2.5f);
		renderer->showGround = true;

	}
	app.renderWindow()->setShadowMultiplier(1.0f);
	app.renderWindow()->setShadowBrightness(0.02f);
	app.renderWindow()->setSamplePower(3.27f);
	app.renderWindow()->setShadowContrast(3.90f);
	app.renderWindow()->getRenderEngine()->shadowQuality = 2048;
	app.renderWindow()->getRenderEngine()->bUseSceneBoundForShadow = true;
	app.renderWindow()->getRenderEngine()->updateShadowMapAttribute();

	app.renderWindow()->getCamera()->setEyePos(Vec3f(0.58, 0.13, 0.28));
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(-0.081, 0.089, -0.035));

	app.mainLoop();

	return 0;
}
