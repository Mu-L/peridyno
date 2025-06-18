#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(1.0, 0, 0);
	BoxInfo box;
	box.halfLength = 0.5f * Vec3f(0.065, 0.065, 0.1);
	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			rigidBody.position = 0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.21 * j + 0.1 * (8 - i));
			rigid->addBox(box, rigidBody);
		}

	SphereInfo sphere;
	sphere.center = Vec3f(0.5f, 0.75f, 0.5f);
	sphere.radius = 0.025f;

	RigidBodyInfo rigidSphere;
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.95f, 0.5f);
	sphere.radius = 0.025f;
	rigid->addSphere(sphere, rigidSphere);

	rigidSphere.position = Vec3f(0.5f, 0.65f, 0.5f);
	sphere.radius = 0.05f;
	rigid->addSphere(sphere, rigidSphere);

	TetInfo tet;
	tet.v[0] = Vec3f(0.5f, 1.1f, 0.5f);
	tet.v[1] = Vec3f(0.5f, 1.2f, 0.5f);
	tet.v[2] = Vec3f(0.6f, 1.1f, 0.5f);
	tet.v[3] = Vec3f(0.5f, 1.1f, 0.6f);

	RigidBodyInfo rigidTet;
	rigidTet.position = (tet.v[0] + tet.v[1] + tet.v[2] + tet.v[3]) / 4;
	tet.v[0] -= rigidTet.position;
	tet.v[1] -= rigidTet.position;
	tet.v[2] -= rigidTet.position;
	tet.v[3] -= rigidTet.position;
	rigid->addTet(tet, rigidTet);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLWireframeVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->varRenderMode()->getDataPtr()->setCurrentKey(GLWireframeVisualModule::LINE);
	mapper->outTriangleSet()->connect(sRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	GlfwApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


