#include "initializeRigidBodyGUI.h"
#include "PConsoleWidget.h"
#include "RigidBody/MultiBodyTuple.h"
#include "NodeFactory.h"
#include "ViewerItem/PTextureMeshViewerWidget.h"
#include "Topology/TextureMeshInterface.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QRegularExpression>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "Node.h"
#include "QtGUI/PPropertyWidget.h"
#include "RigidBody/ConfigurableBody.h"
#include "FBXLoader/FBXLoader.h"
#include "GltfLoader.h"
#include "TextureMeshLoader.h"
#include "Topology/TextureMeshInterface.h"
#include "QtGUI/ViewerItem/GLMeshRenderWidget.h"


namespace dyno
{
	// Forward declare callback functions (defined below)
	static void openMeshCallback(const QString& filePath, QWidget* parent);
	static void createRigidBodyCallback(const QString& filePath, QWidget* parent);

	std::atomic<RigidBodyGUIInitializer*> RigidBodyGUIInitializer::gInstance;
	std::mutex RigidBodyGUIInitializer::gMutex;
	std::vector<std::shared_ptr<Node>> RigidBodyGUIInitializer::assetNodes;

	PluginEntry* RigidBodyGUIInitializer::instance()
	{
		RigidBodyGUIInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new RigidBodyGUIInitializer();
				ins->setName("RigidBodyGUI");
				ins->setVersion("1.0");
				ins->setDescription("Rigid body GUI extensions for QtGUI");

				gInstance.store(ins, std::memory_order_release);
			}
		}
		return ins;
	}

	void RigidBodyGUIInitializer::initializeActions()
	{
		// Register "Open" and "Create Rigid Body" for common 3D model file types
		std::vector<std::string> modelExts = { "obj", "gltf", "glb", "fbx", "stl" };
		for (const std::string& ext : modelExts)
		{
			QContentBrowser::registerAction(ext, "Open", openMeshCallback);
			QContentBrowser::registerAction(ext, "Create Rigid Body", createRigidBodyCallback);
		}
	}

	/**
	 * "Open" action: load the mesh file and show it in a render window
	 */
	static void openMeshCallback(const QString& filePath, QWidget* parent)
	{
		QFileInfo fileInfo(filePath);
		std::string ext = fileInfo.suffix().toStdString();

		auto ext2Act = NodeFactory::instance()->nodeContentActions();
		if (ext2Act.find(ext) == ext2Act.end())
		{
			QMessageBox::warning(parent, "Warning",
				"No loader registered for file type: " + fileInfo.suffix());
			return;
		}

		auto func = ext2Act[ext];
		if (!func)
			return;

		auto node = func(filePath.toStdString());
		if (!node)
			return;

		RigidBodyGUIInitializer::assetNodes.push_back(node);
		auto texMeshInterface = std::dynamic_pointer_cast<TextureMeshInterface>(node);
		if (!texMeshInterface)
		{
			QMessageBox::warning(parent, "Warning", "This file type does not support TextureMesh preview.");
			return;
		}

		auto field = texMeshInterface->getTextureMesh();
		if (!field)
			return;

		QWidget* renderWindow = new QWidget();
		renderWindow->setWindowTitle("Mesh Viewer - " + fileInfo.fileName());
		renderWindow->setMinimumSize(800, 600);

		QVBoxLayout* layout = new QVBoxLayout(renderWindow);
		layout->setContentsMargins(0, 0, 0, 0);

		PTextureMeshViewerWidget* viewer = new PTextureMeshViewerWidget(field, renderWindow);
		layout->addWidget(viewer);

		renderWindow->setLayout(layout);
		renderWindow->setAttribute(Qt::WA_DeleteOnClose);
		renderWindow->show();
	}

	/**
	 * "Create Rigid Body" action: show a config dialog and generate a .pdm file
	 */
	static void createRigidBodyCallback(const QString& filePath, QWidget* parent)
	{
		QFileInfo fileInfo(filePath);

		// --- Configuration Dialog ---
		QDialog* dialog = new QDialog(parent);
		dialog->setWindowTitle("Create Rigid Body Configuration");
		dialog->setMinimumSize(800, 600);
		dialog->setAttribute(Qt::WA_DeleteOnClose);

		QHBoxLayout* mainLayout = new QHBoxLayout(dialog);
		QVBoxLayout* sideLayout = new QVBoxLayout();
		mainLayout->addLayout(sideLayout);

		QLabel* fileLabel = new QLabel("Source file: " + filePath);
		sideLayout->addWidget(fileLabel);

		PPropertyWidget* propertyWidget = new PPropertyWidget();
		propertyWidget->setMinimumWidth(100);

		auto renderWidget = new GLMeshRenderWidget();
		renderWidget->setMinimumSize(400, 300);
		mainLayout->addWidget(renderWidget, 1);

		auto configNode = std::make_shared<ConfigurableBody<DataType3f>>();
		QObject::connect(propertyWidget, &PPropertyWidget::nodeUpdated, renderWidget, &GLMeshRenderWidget::onNodeUpdated);

		auto ext = fileInfo.suffix().toStdString();
		std::shared_ptr<Node> fileNode;

		if (ext == std::string("fbx"))
		{
			auto fbxNode = std::make_shared<FBXLoader<DataType3f>>();
			fileNode = fbxNode;
			fbxNode->varFileName()->setValue(filePath.toStdString());
			fbxNode->stateTextureMesh()->connect(configNode->inTextureMesh());
		}
		else if (ext == std::string("obj")) 
		{
			auto objNode = std::make_shared<TextureMeshLoader>();
			fileNode = objNode;
			objNode->varFileName()->setValue(filePath.toStdString());
			objNode->stateTextureMesh()->connect(configNode->inTextureMesh());
		}
		else if (ext == std::string("gltf")|| ext == std::string("glb"))
		{
			auto gltfNode = std::make_shared<GltfLoader<DataType3f>>();
			fileNode = gltfNode;
			gltfNode->varFileName()->setValue(filePath.toStdString());
			gltfNode->stateTextureMesh()->connect(configNode->inTextureMesh());
		}

		// Add node to scene for processing
		renderWidget->addNode(fileNode);
		renderWidget->addNode(configNode);
		auto repaintWidget = [=]() {
			renderWidget->update();
		};


		auto texMeshInterface = std::dynamic_pointer_cast<TextureMeshInterface>(fileNode);
		if (texMeshInterface) 
		{
			if (auto texMeshField = texMeshInterface->getTextureMesh()) 
			{
				//renderWidget->addField(texMeshField);
				//QGridLayout* varlayout = new QGridLayout();
				//QVBoxLayout* propertyLayout = new QVBoxLayout();
				//propertyWidget->setLayout(propertyLayout);
				//propertyLayout->addLayout(varlayout);
				//auto configWidget = propertyWidget->addVariableFieldWidget(configNode->varConfiguration(), varlayout);
				//propertyLayout->addWidget(configWidget);
				propertyWidget->showNodeProperty(configNode);
				sideLayout->addWidget(propertyWidget, 1);




				QObject::connect(propertyWidget, QOverload<std::shared_ptr<Node>>::of(&PPropertyWidget::nodeUpdated), repaintWidget);


			}

		}
		else 
		{
			printf("%s", "RIGIDBODYGUI :: ERROR SOURCE FILE!!\n");
		}

		QHBoxLayout* buttonLayout = new QHBoxLayout();
		sideLayout->addLayout(buttonLayout);
		QPushButton* resetScene = new QPushButton("Reset Scene");
		sideLayout->addStretch();
		sideLayout->addWidget(resetScene);
		auto conn = QObject::connect(resetScene, &QPushButton::released,
			[=]() {

				renderWidget->resetScene();
				renderWidget->update();
				printf("[Reset RigidBodyGUI Scene]\n");
			}
		);

		renderWidget->update();

		dialog->exec();

	}
}

// Static plugin initialization (called when statically linked)
dyno::PluginEntry* RigidBodyGUI::initStaticPlugin()
{
	if (dyno::RigidBodyGUIInitializer::instance()->initialize())
		return dyno::RigidBodyGUIInitializer::instance();

	return nullptr;
}

// Dynamic plugin entry point (called by PluginManager::loadPlugin)
PERIDYNO_API dyno::PluginEntry* RigidBodyGUI::initDynoPlugin()
{
	if (dyno::RigidBodyGUIInitializer::instance()->initialize())
		return dyno::RigidBodyGUIInitializer::instance();

	return nullptr;
}

// Auto-initialization: this static object ensures registration happens when DLL is loaded
namespace
{
	struct AutoInit
	{
		AutoInit()
		{
			// Automatically initialize when DLL loads
			RigidBodyGUI::initStaticPlugin();
		}
	};

	static AutoInit gAutoInit;
}