#include "UbiApp.h"
#include <SceneGraph.h>

#include <BasicShapes/PlaneModel.h>
#include <RigidBody/Gear.h>
#include <RigidBody/MultibodySystem.h>
#include "RigidBody/ConfigurableBody.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createSceneGraph()
{
	auto scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(24.0f);
	plane->varLengthZ()->setValue(12.0f);
	plane->varSegmentX()->setValue(24);
	plane->varSegmentZ()->setValue(12);

	auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	//auto scene = scn->addNode(std::make_shared<MatBody<DataType3f>>());
	//scene->varFrictionCoefficient()->setValue(0.35f);
	//scene->setXMLPath("ma/scene_ragdoll_mesh_stack.xml");
	//scene->connect(convoy->importVehicles());

	auto config = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	config->varLoadConfigPath()->setValue(FilePath(getAssetPath() + "ma/scene_chain_pyramid_impact_box_mass_x5.pdm"));
	config->varFrictionCoefficient()->setValue(0.35f);

	config->connect(convoy->importVehicles());

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}
