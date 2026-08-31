#include <UbiApp.h>
#include <QtApp.h>
#include <GlfwGUI/GlfwApp.h>

#include <SceneGraph.h>

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"
#include "RigidBody/Vehicle.h"

#include "Module/KeyboardInputModule.h"
#include <UbiGUI/UbiApp.h>
#include "RigidBody/ConfigurableBody.h"
#include "FBXLoader/FBXLoader.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatArm()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto fbx = scn->addNode(std::make_shared<FBXLoader<DataType3f>>());
	fbx->varFileName()->setValue(std::string("C:/Users/win11/Desktop/LODTest/Arm_0.fbx"));
	fbx->varLOD1()->setValue(std::string("C:/Users/win11/Desktop/LODTest/Arm_1.fbx"));
	fbx->varLOD2()->setValue(std::string("C:/Users/win11/Desktop/LODTest/Arm_2.fbx"));
	fbx->varUseInstanceTransform()->setValue(true);
	fbx->setVisible(false);
	fbx->reset();

	auto arm = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	fbx->stateTextureMesh()->connect(arm->inTextureMesh());
	arm->varLocation()->setValue(Vec3f(0, 0.3, 0));
	arm->varVehiclesTransform()->clear();

	for (size_t i = 0; i < 700; i++)
	{
		int rowElementNum = 28;
		arm->varVehiclesTransform()->pushBack(Transform3f(Vec3f(float(i)*1.2f/(float)rowElementNum,0, (float)(i%rowElementNum) * 1.2),Mat3f::identityMatrix()));
	}

	auto multisystem = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	std::string arm6 = std::string("Model::Arm6");
	std::string arm5 = std::string("Model::Arm5");
	std::string arm4 = std::string("Model::Arm4");
	std::string arm3 = std::string("Model::Arm3");
	std::string arm2 = std::string("Model::Arm2");
	std::string arm1 = std::string("Model::Arm1");
	std::string arm0 = std::string("Model::Arm0");
	std::string twist = std::string("Model::Twist");
	std::string L_Finger4 = std::string("Model::L_Finger4");
	std::string L_Finger3 = std::string("Model::L_Finger3");
	std::string L_Finger2 = std::string("Model::L_Finger2");
	std::string L_FingerTip1 = std::string("Model::L_FingerTip1");
	std::string R_Finger4 = std::string("Model::R_Finger4");
	std::string R_Finger3 = std::string("Model::R_Finger3");
	std::string R_Finger2 = std::string("Model::R_Finger2");
	std::string R_FingerTip1 = std::string("Model::R_FingerTip1");

	MultiBodyTuple multiBodyConfig;


	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm6, 6, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm6), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm5, 1, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm5), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm4, 2, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm4), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm3, 3, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm3), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm2, 4, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm2), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm1, 5, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm1), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(arm0, 6, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(arm0), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(twist, 7, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(twist), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(L_Finger4, 8, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(L_Finger4), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(L_Finger3, 9, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(L_Finger3), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(L_Finger2, 10, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(L_Finger2), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(L_FingerTip1, 11, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(L_FingerTip1), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(R_Finger4, 12, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(R_Finger4), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(R_Finger3, 13, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(R_Finger3), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(R_Finger2, 14, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(R_Finger2), RigidShapeType::SHAPE_BOX, 100));
	multiBodyConfig.varRigidBodyConfigs()->pushBack(RigidBodyTuple(R_FingerTip1, 15, fbx->stateHierarchicalScene()->getDataPtr()->findMeshIndexByName(R_FingerTip1), RigidShapeType::SHAPE_BOX, 100));

	int i = 0;
	for (auto it = multiBodyConfig.varRigidBodyConfigs()->begin(); it != multiBodyConfig.varRigidBodyConfigs()->end(); it++,i++)
	{
		auto rigid = multiBodyConfig.varRigidBodyConfigs()->getElement(it);
		auto shape = rigid.varShapeConfigs()->getElement(rigid.varShapeConfigs()->begin());
		shape.varRadius()->setValue(0.4);

		auto fieldPtr = multiBodyConfig.varRigidBodyConfigs()->get(0);
		auto rigidPtr = dynamic_cast<TFTuple<RigidBodyTuple>*>(fieldPtr);
		if (rigidPtr) 
		{
			rigidPtr->constDataPtr()->varConfigGroup()->setValue(0);
		}
	}
	
	auto field0 = multiBodyConfig.varRigidBodyConfigs()->get(0);
	auto rigid0 = dynamic_cast<TFTuple<RigidBodyTuple>*>(field0);
	if(rigid0)
		rigid0->constDataPtr()->varMotionType()->setCurrentKey(RigidMotionType::RIGID_Static);


	float vScale = 0.01;
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(arm5,  arm6, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(19.461, -4.84, 51.043) * vScale, false, 0, false, 0, 0, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(arm4,  arm5, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(11.15, 3.774, 51.043) * vScale, false, 0, false, 0, 0, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(arm3, arm4, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(10.943, 3.693, -0.181) * vScale, false, 0, false, 0, 0, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(arm2, arm3, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(11.971, 3.792, -47.372) * vScale, false, 0, false, 0, 0, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(arm1, arm2, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(6.392, -1.914, -47.391) * vScale, false, 0, false, 0, 0, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(arm0, arm1, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0.616, -7.439, -47.391) * vScale, false, 0, false, 0, 0, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(twist, arm0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(-3.519, -7.485, -47.355) * vScale, false, 0, true, 0, 0.0001, false));

	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(L_Finger4,  twist, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-10.072, -7.485, -43.111) * vScale, false, 0, true, 0, 0.0001, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(R_Finger4,  twist, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-10.072, -7.485, -51.711) * vScale, false, 0, true, 0, 0.0001, false));

	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(L_Finger3, L_Finger4, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-9.572, -6.833, -39.908) * vScale, false, 0, true, 0, 0.0001, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(R_Finger3,  R_Finger4, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-9.572, -6.833, -54.899) * vScale, false, 0, true, 0, 0.0001, false));

	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(L_Finger2, twist, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-10.91, -5.286, -45.854) * vScale, false, 0, true, 0, 0.0001, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(R_Finger2, twist, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-10.91, -5.286, -48.911) * vScale, false, 0, true, 0, 0.0001, false));

	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(L_FingerTip1, L_Finger3, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-19.087, -6.423, -35.64) * vScale, false, 0, true, 0, 0.0001, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(R_FingerTip1, R_Finger3, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-19.087, -6.424, -59.169) * vScale, false, 0, true, 0, 0.0001, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(L_FingerTip1, L_Finger2, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-19.874, -5.762, -37.807) * vScale, false, 0, true, 0, 0.0001, false));
	multiBodyConfig.varJointConfigs()->pushBack(MultiBodyJointTuple(R_FingerTip1, R_Finger2, JointType::JOINT_Hinge, Vec3f(0, 1, 0), Vec3f(-19.874, -5.762, -57.002) * vScale, false, 0, true, 0, 0.0001, false));


	arm->varConfiguration()->setValue(multiBodyConfig);

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	arm->connect(multisystem->importVehicles());
	plane->stateTriangleSet()->connect(multisystem->inTriangleSet());
	plane->varLengthX()->setValue(120);
	plane->varLengthZ()->setValue(120);
	plane->varLocation()->setValue(Vec3f(0,-0.5,0));

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(creatArm());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	//app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
