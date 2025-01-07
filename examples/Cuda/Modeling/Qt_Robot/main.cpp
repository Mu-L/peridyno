#include <QtApp.h>

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "Peridynamics/initializePeridynamics.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "Volume/initializeVolume.h"
#include "Multiphysics/initializeMultiphysics.h"
#include "initializeModeling.h"
#include "initializeIO.h"
#include "GltfLoader.h"
#include <GLRenderEngine.h>
#include "JointDeform.h"
#include "SkeletonLoader/SkeletonLoader.h"
#include "RigidBody/Module/AnimationDriver.h"
#include "RigidBody/Module/CarDriver.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"

#include "RigidBody/ConfigurableBody.h"


/**
 * @brief This example demonstrate how to load plugin libraries in a static way
 */
using namespace dyno;
int main()
{

	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto fbx = scn->addNode(std::make_shared<SkeletonLoader<DataType3f>>());
	fbx->varFileName()->setValue(getAssetPath() + "fbx/HumanoidRobot.fbx");
	fbx->reset();

	auto robot = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	fbx->stateTextureMesh()->connect(robot->inTextureMesh());
	fbx->setVisible(false);


	VehicleBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);

	std::string Hip = std::string("model:Hip");

	std::string Trochanter_R = std::string("model:Trochanter_R");
	std::string Thigh_R = std::string("model:Thigh_R");
	std::string Shank_R = std::string("model:Shank_R");
	std::string Foot_R = std::string("model:Foot_R");

	std::string Trochanter_L = std::string("model:Trochanter_L");
	std::string Thigh_L = std::string("model:Thigh_L");
	std::string Shank_L = std::string("model:Shank_L");
	std::string Foot_L = std::string("model:Foot_L");

	std::string Spine = std::string("model:Spine");
	std::string Body = std::string("model:Body");
	std::string Neck = std::string("model:Neck");
	std::string Head = std::string("model:Head");

	std::string Shoulder_R = std::string("model:Shoulder_R");
	std::string UpperArm_R = std::string("model:UpperArm_R");
	std::string LowerArm_R = std::string("model:LowerArm_R");
	std::string Wrist_R = std::string("model:Wrist_R");
	std::string Hand_R = std::string("model:Hand_R");

	std::string Shoulder_L = std::string("model:Shoulder_L");
	std::string UpperArm_L = std::string("model:UpperArm_L");
	std::string LowerArm_L = std::string("model:LowerArm_L");
	std::string Wrist_L = std::string("model:Wrist_L");
	std::string Hand_L = std::string("model:Hand_L");


	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Hip, 0), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hip), Box, Transform3f(), 100));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Trochanter_R, 1), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Trochanter_R), Box, Transform3f(), 100));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Thigh_R, 2), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Thigh_R), Box, Transform3f(), 100));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Shank_R, 3), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shank_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Foot_R, 4), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Foot_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Trochanter_L, 5), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Trochanter_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Thigh_L, 6), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Thigh_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Shank_L, 7), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shank_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Foot_L, 8), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Foot_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Spine, 9), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Spine), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Body, 10), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Body), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Neck, 11), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Neck), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Head, 12), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Head), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Shoulder_R, 13), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shoulder_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(UpperArm_R, 14), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(UpperArm_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(LowerArm_R, 15), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(LowerArm_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Wrist_R, 16), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Wrist_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Hand_R, 17), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hand_R), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Shoulder_L, 18), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Shoulder_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(UpperArm_L, 19), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(UpperArm_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(LowerArm_L, 20), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(LowerArm_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Wrist_L, 21), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Wrist_L), Box, Transform3f(), 100));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape(Hand_L, 22), fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(Hand_L), Box, Transform3f(), 100));

	for (size_t i = 0; i < configData.mVehicleRigidBodyInfo.size(); i++)
	{
		configData.mVehicleRigidBodyInfo[i].radius = 0.2;
	}

	Vec3f offset = Vec3f(0, 0, 0);
	Vec3f shankOffset = Vec3f(0, 0.25, 0);
	Vec3f thighOffset = Vec3f(0, 0.28, 0);
	Vec3f footOffset = Vec3f(0, 0, 0.02);
	Vec3f bodyOffset = Vec3f(0, -0.25, 0);
	Vec3f handOffset = Vec3f(0.02, 0, 0);

	Vec3f axis = Vec3f(-1,0,0);

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Trochanter_R, 1), Name_Shape(Hip, 0), Hinge, axis, offset, true, 0,true,-90,90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Trochanter_L, 5), Name_Shape(Hip, 0), Hinge, axis, offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Thigh_R, 2), Name_Shape(Hip, 0), Hinge, axis, thighOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Thigh_L, 6), Name_Shape(Hip, 0), Hinge, axis, thighOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Shank_R, 3), Name_Shape(Thigh_R, 2), Hinge, axis, shankOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Shank_L, 7), Name_Shape(Thigh_L, 6), Hinge, axis, shankOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Foot_R, 4), Name_Shape(Shank_R, 3), Hinge, axis, footOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Foot_L, 8), Name_Shape(Shank_L, 7), Hinge, axis, footOffset, true, 0, true, -90, 90));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Spine, 9), Name_Shape(Hip, 0), Fixed, Vec3f(0, 1, 0), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Body, 10), Name_Shape(Spine, 9), Fixed, Vec3f(1, 0, 0), bodyOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Neck, 11), Name_Shape(Body, 10), Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Head, 12), Name_Shape(Neck, 11), Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Shoulder_R, 13), Name_Shape(Body, 10), Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Shoulder_L, 18), Name_Shape(Body, 10), Fixed, Vec3f(1, 0, 0), offset, true, 0, true, -90, 90));
	
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(UpperArm_R, 14), Name_Shape(Shoulder_R, 13), Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(UpperArm_L, 19), Name_Shape(Shoulder_L, 18), Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(LowerArm_R, 15), Name_Shape(UpperArm_R, 14), Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(LowerArm_L, 20), Name_Shape(UpperArm_L, 19), Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Wrist_R, 16), Name_Shape(LowerArm_R, 15), Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Wrist_L, 21), Name_Shape(LowerArm_L, 20), Fixed, Vec3f(0, 0, 1), offset, true, 0, true, -90, 90));
	
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Hand_R, 17), Name_Shape(Wrist_R, 16), Fixed, Vec3f(1, 0, 0), handOffset, true, 0, true, -90, 90));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape(Hand_L, 22), Name_Shape(Wrist_L, 21), Fixed, Vec3f(1, 0, 0), -handOffset, true, 0, true, -90, 90));

	robot->varVehicleConfiguration()->setValue(configData);

	robot->varGravityValue()->setValue(0);
	
	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	robot->connect(multibody->importVehicles());

	auto animDriver = std::make_shared<AnimationDriver<DataType3f>>();
	multibody->animationPipeline()->pushModule(animDriver);

	std::vector<std::string> driveName(8, "NULL");

	driveName[0] = std::string("Hip_R");
	driveName[1] = std::string("Hip_L");
	driveName[2] = std::string("Hip_R");
	driveName[3] = std::string("Hip_L");

	driveName[4] = std::string("Knee_R");
	driveName[5] = std::string("Knee_L");
	driveName[6] = std::string("Ankle_R");
	driveName[7] = std::string("Ankle_L");



	animDriver->varDriverName()->setValue(driveName);
	animDriver->varSpeed()->setValue(1);
	multibody->stateTimeStep()->connect(animDriver->inDeltaTime());
	fbx->stateHierarchicalScene()->connect(animDriver->inHierarchicalScene());
	multibody->stateTopology()->connect(animDriver->inTopology());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(20));
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());


	Modeling::initStaticPlugin();
	RigidBody::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	Peridynamics::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();
	Volume::initStaticPlugin();
	Multiphysics::initStaticPlugin();
	dynoIO::initStaticPlugin();


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1280	, 720);

	// setup envmap
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
	}

	app.mainLoop();

	return 0;
}