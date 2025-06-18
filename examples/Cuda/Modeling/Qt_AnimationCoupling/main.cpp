#include <UbiApp.h>
#include <SceneGraph.h>
#include <GLRenderEngine.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>
#include <ParticleSystem/Module/ParticleIntegrator.h>
#include <ParticleSystem/Module/IterativeDensitySolver.h>
#include <ParticleSystem/Module/ImplicitViscosity.h>

#include <Collision/NeighborPointQuery.h>

//Rendering
#include <GLSurfaceVisualModule.h>
#include <GLPhotorealisticInstanceRender.h>

#include <Commands/Merge.h>

#include <BasicShapes/CubeModel.h>
#include <Samplers/ShapeSampler.h>

#include <Node/GLPointVisualNode.h>

#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

#include <ColorMapping.h>
#include <Module/CalculateNorm.h>

#include <GltfLoader.h>

#include "Auxiliary/DataSource.h"

#include <RigidBody/Vehicle.h>
#include <RigidBody/MultibodySystem.h>
#include <RigidBody/Module/InstanceTransform.h>

#include <Mapping/TextureMeshToTriangleSet.h>
#include <Mapping/MergeTriangleSet.h>
#include "FBXLoader/FBXLoader.h"

using namespace dyno;

class GenerateInstances : public Node
{
public:
	GenerateInstances() {
		this->stateTransform()->allocate();
	};

	void resetStates() override
	{
		auto mesh = this->inTextureMesh()->constDataPtr();
		const int instanceCount = 1;
		const int shapeNum = mesh->shapes().size();

		std::vector<std::vector<Transform3f>> transform(shapeNum);

		for (size_t j = 0; j < instanceCount; j++)
		{
			for (size_t i = 0; i < shapeNum; i++) {

				auto shapeTransform = this->inTextureMesh()->constDataPtr()->shapes()[i]->boundingTransform;

				transform[i].push_back(Transform3f(shapeTransform.translation(), shapeTransform.rotation(), shapeTransform.scale()));
			}
		}

		auto tl = this->stateTransform()->getDataPtr();
		tl->assign(transform);
	}

	//DEF_VAR(Vec3f, Offest, Vec3f(0.4, 0, 0), "");

	DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
	DEF_ARRAYLIST_STATE(Transform3f, Transform, DeviceType::GPU, "");
};


std::shared_ptr<SceneGraph> creatScene();
void importOtherModel(std::shared_ptr<SceneGraph> scn);

float total_scale = 8;

std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setAsynchronousSimulation(false);

	//***************************************Scene Setting***************************************//
	// Scene Setting
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-0.5f, 0.0f, -4.0f) * total_scale);
	scn->setUpperBound(Vec3f(0.5f, 1.0f, 4.0f) * total_scale);


	auto dancing = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	dancing->varFileName()->setValue(getAssetPath() + "fbx/SwingDancing.fbx");
	dancing->varUseInstanceTransform()->setValue(false);
	dancing->varImportAnimation()->setValue(true);
	dancing->varLocation()->setValue(Vec3f(0.1, 0, -0.5));
	dancing->varScale()->setValue(Vec3f(1.8));

	auto tex2TriSet = scn->addNode(std::make_shared<TextureMeshToTriangleSetNode<DataType3f>>());
	dancing->stateTextureMesh()->connect(tex2TriSet->inTextureMesh());


	//*************************************** Cube Sample ***************************************//
	// Cube 
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0,0.35,-0.6) );
	cube->varLength()->setValue(Vec3f(2.8, 0.7, 6));
	cube->varScale()->setValue(Vec3f(2, 1, 0.932));
	cube->graphicsPipeline()->disable();

	auto cubeSmapler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	cubeSmapler->varSamplingDistance()->setValue(0.004f * total_scale);
	cube->connect(cubeSmapler->importShape());
	cubeSmapler->graphicsPipeline()->disable();

	//MakeParticleSystem
	auto particleSystem = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	cubeSmapler->statePointSet()->promoteOuput()->connect(particleSystem->inPoints());

	//*************************************** Fluid ***************************************//
	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->setDt(0.004f);

	{
		fluid->animationPipeline()->clear();

		auto smoothingLength = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		smoothingLength->varValue()->setValue(0.006f * total_scale);

		auto samplingDistance = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		samplingDistance->varValue()->setValue(Real(0.004) * total_scale);

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		fluid->stateTimeStep()->connect(integrator->inTimeStep());
		fluid->statePosition()->connect(integrator->inPosition());
		fluid->stateVelocity()->connect(integrator->inVelocity());
		fluid->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
		fluid->statePosition()->connect(nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(nbrQuery);

		auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
		density->varKappa()->setValue(0.1f);

		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);

		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());

		auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
		viscosity->varViscosity()->setValue(Real(10.0));
		fluid->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		samplingDistance->outFloating()->connect(viscosity->inSamplingDistance());
		fluid->statePosition()->connect(viscosity->inPosition());
		fluid->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		fluid->animationPipeline()->pushModule(viscosity);

		auto pointRender = fluid->graphicsPipeline()->findFirstModule<GLPointVisualModule>();
		if (pointRender != nullptr)
			pointRender->varPointSize()->setValue(0.015f);
	}

	particleSystem->connect(fluid->importInitialStates());

	//TriangularMeshBoundary
	auto meshBoundary = scn->addNode(std::make_shared<TriangularMeshBoundary<DataType3f>>());
	meshBoundary->varThickness()->setValue(0.005f * total_scale);

	fluid->connect(meshBoundary->importParticleSystems());
	tex2TriSet->outTriangleSet()->connect(meshBoundary->inTriangleSet());

	//Create a boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 3.006f, -0.6));
	cubeBoundary->varScale()->setValue(Vec3f(2, 1, 0.932));
	cubeBoundary->varLength()->setValue(Vec3f(2.9f, 6.0f, 6.200f));


	auto glSurface = cubeBoundary->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glSurface->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.1f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());

	fluid->connect(container->importParticleSystems());

	cubeBoundary->setVisible(true);

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);
	window.setSceneGraph(creatScene());
	window.initialize(1366, 768);

	//Set the distance unit for the camera, the fault unit is meter
	window.renderWindow()->getCamera()->setUnitScale(3.0f);

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(window.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);

	}

	window.mainLoop();

	return 0;
}