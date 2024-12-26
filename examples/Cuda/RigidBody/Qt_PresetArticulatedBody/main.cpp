#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>
#include <RigidBody/MultibodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <BasicShapes/PlaneModel.h>

#include <Collision/NeighborElementQuery.h>
#include "RigidBody/PresetArticulatedBody.h"

using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> createSceneGraph()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto jeep = scn->addNode(std::make_shared<PresetJeep<DataType3f>>());
	jeep->varLocation()->setValue(Vec3f(4,0,0));
	auto tank = scn->addNode(std::make_shared<PresetTank<DataType3f>>());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(50);
	plane->varLengthZ()->setValue(50);
	plane->varSegmentX()->setValue(10);
	plane->varSegmentZ()->setValue(10);

	auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	//jeep->connect(convoy->importVehicles());
	//tank->connect(convoy->importVehicles());

	plane->stateTriangleSet()->connect(jeep->inTriangleSet());
	plane->stateTriangleSet()->connect(tank->inTriangleSet());

	plane->stateTriangleSet()->connect(convoy->inTriangleSet());

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());

	app.initialize(1280, 768);
	app.renderWindow()->getCamera()->setUnitScale(3.0f);
	app.mainLoop();

	return 0;
}