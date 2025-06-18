#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rA;
	SphereInfo box;
	box.center = Vec3f(0.0f);
	//box.halfLength = 2.0 * Vec3f(0.02, 0.08, 0.02);
	//box.halfLength = 0.1;
	box.radius = 0.15;
	box.rot = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
	rA.position = 2.0 * Vec3f(-1.0, 0.2, 0.5);
	rA.linearVelocity = Vec3f(1, 0, 0);
	auto oldBoxActor = rigid->addSphere(box, rA);
	rA.linearVelocity = Vec3f(0, 0, 0);

	for (int i = 0; i < 60; i++)
	{
		RigidBodyInfo rB;
		rB.position = rA.position + 2.0 * Vec3f(0.0, 0.2, 0);
		rB.bodyId = i + 1;

		auto newBoxActor = rigid->addSphere(box, rB);
		auto& hingeJoint = rigid->createHingeJoint(oldBoxActor, newBoxActor);
		hingeJoint.setAnchorPoint((rA.position + rB.position) / 2);
		hingeJoint.setAxis(Vec3f(0, 0, 1));
		hingeJoint.setRange(-M_PI/2, M_PI/2);

		rA = rB;
		oldBoxActor = newBoxActor;
	}
	
	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(1.0f);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactMapper->inContacts());
	contactMapper->varScale()->setValue(0.02);
	rigid->graphicsPipeline()->pushModule(contactMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 0, 1));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Color(1, 0, 0));
	pointRender->varPointSize()->setValue(0.003f);
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


