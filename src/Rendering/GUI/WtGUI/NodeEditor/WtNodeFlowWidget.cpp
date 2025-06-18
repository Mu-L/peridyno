#include "WtNodeFlowWidget.h"
#include <Wt/WEnvironment.h>
#include <Wt/WApplication.h>
#include <Wt/WMessageBox.h>

WtNodeFlowWidget::WtNodeFlowWidget(std::shared_ptr<dyno::SceneGraph> scene)
	: WtFlowWidget(scene)
{
	setPreferredMethod(Wt::RenderMethod::HtmlCanvas);

	this->mouseWentDown().connect(this, &WtNodeFlowWidget::onMouseWentDown);
	this->mouseMoved().connect(this, &WtNodeFlowWidget::onMouseMove);
	this->mouseWentUp().connect(this, &WtNodeFlowWidget::onMouseWentUp);
}

WtNodeFlowWidget::~WtNodeFlowWidget() {};

void WtNodeFlowWidget::onMouseWentDown(const Wt::WMouseEvent& event)
{
	isDragging = true;
	mLastMousePos = Wt::WPointF(event.widget().x, event.widget().y);
	mLastDelta = Wt::WPointF(0, 0);
	if (!checkMouseInAllRect(Wt::WPointF(event.widget().x, event.widget().y)))
	{
		selectType = -1;
		selectedNum = 0;
		canMoveNode = false;
		update();

	}
	if (selectType > 0)
	{
		auto origin = nodeMap[selectedNum]->flowNodeData().getNodeOrigin();
		mTranslateNode = Wt::WPointF(origin.x(), origin.y());
		if (checkMouseInPoints(mLastMousePos, nodeMap[selectedNum]->flowNodeData(), PortState::out))
		{
			drawLineFlag = true;

			if (nodeMap.find(selectedNum) != nodeMap.end())
			{
				mOutNode = nodeMap.find(selectedNum)->second->getNode();
			}

			if (outPoint.portType == PortType::In)
			{
				auto existConnection = nodeMap[selectedNum]->getIndexConnection(outPoint.portIndex);
				if (existConnection != nullptr)
				{
					auto outNode = existConnection->getNode(PortType::Out);
					for (auto it = mScene->begin(); it != mScene->end(); it++)
					{
						auto m = it.get();
						auto node = nodeMap[m->objectId()];
						auto outPortIndex = existConnection->getPortIndex(PortType::Out);
						auto exportPortsData = outNode->flowNodeData().getPointsData();
						connectionPointData exportPointData;
						for (auto point : exportPortsData)
						{
							if (point.portIndex == outPortIndex)
							{
								exportPointData = point;
								break;
							}
						}

						if (node == outNode)
						{
							for (auto it = sceneConnections.begin(); it != sceneConnections.end(); )
							{
								if (it->exportNode == mOutNode && it->inportNode == m && it->inPoint.portIndex == outPoint.portIndex && it->outPoint.portIndex == exportPointData.portIndex)
								{
									it = sceneConnections.erase(it);
								}
								else
								{
									++it;
								}
							}

							disconnect(m, mOutNode, outPoint, exportPointData, nodeMap[selectedNum], outNode);
							sourcePoint = getPortPosition(outNode->flowNodeData().getNodeOrigin(), exportPointData);

						}
					}

				}
			}
		}
		else
		{
			// selectType = 2: selected & drag
			// selectType = 1: mouse move node
			// selectType = -1: no select
			selectType = 2;
		}
	}

}

void WtNodeFlowWidget::onMouseMove(const Wt::WMouseEvent& event)
{
	sinkPoint = Wt::WPointF(event.widget().x / mZoomFactor - mTranslate.x(), event.widget().y / mZoomFactor - mTranslate.y());
	if (isDragging && selectType < 0)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslate = Wt::WPointF(mTranslate.x() + delta.x() - mLastDelta.x(), mTranslate.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
	else if (isDragging && selectType > 0)
	{
		Wt::WPointF delta = Wt::WPointF(event.dragDelta().x, event.dragDelta().y);
		mTranslateNode = Wt::WPointF(mTranslateNode.x() + delta.x() - mLastDelta.x(), mTranslateNode.y() + delta.y() - mLastDelta.y());
		update();
		mLastDelta = delta;
	}
	else
	{
		for (auto it = mScene->begin(); it != mScene->end(); it++)
		{
			auto m = it.get();
			auto node = nodeMap[m->objectId()];
			auto nodeData = node->flowNodeData();
			auto mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
			if (checkMouseInRect(mousePoint, nodeData) && selectType != 2)
			{
				selectType = 1;
				connectionOutNode = node;
				selectedNum = m->objectId();
				canMoveNode = true;
				update();
				break;
			}
		}
	}
}

void WtNodeFlowWidget::onMouseWentUp(const Wt::WMouseEvent& event)
{
	isDragging = false;
	mLastDelta = Wt::WPointF(0, 0);
	mTranslateNode = Wt::WPointF(0, 0);
	Wt::WPointF mouseWentUpPosition = Wt::WPointF(event.widget().x, event.widget().y);
	if (selectType > 0)
	{
		auto node = nodeMap[selectedNum];
		auto nodeData = node->flowNodeData();

		_selectNodeSignal.emit(selectedNum);

		Wt::WPointF mousePoint = Wt::WPointF(event.widget().x, event.widget().y);
		if (!checkMouseInAllRect(mousePoint) && selectType == 2)
		{
			selectType = -1;
			selectedNum = 0;
			canMoveNode = false;
			update();
			_updateCanvas.emit();
		}

		if (checkMouseInHotKey0(mousePoint, nodeData))
		{
			auto nodeWidget = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
			auto m = nodeWidget->getNode();
			if (m->isVisible())
			{
				enableRendering(*node, false);
			}
			else
			{
				enableRendering(*node, true);
			}
			//mMainWindow->updateCanvas();
			_updateCanvas.emit();
			update();
		}

		if (checkMouseInHotKey1(mousePoint, nodeData))
		{
			auto nodeWidget = dynamic_cast<WtNodeWidget*>(node->nodeDataModel());
			auto m = nodeWidget->getNode();
			if (m->isActive())
			{
				enablePhysics(*node, false);
			}
			else
			{
				enablePhysics(*node, true);
			}
			//mMainWindow->updateCanvas();
			_updateCanvas.emit();
			update();
		}
	}
	else
	{
		_selectNodeSignal.emit(-1);
	}

	if (drawLineFlag = true)
	{
		for (auto it = mScene->begin(); it != mScene->end(); it++)
		{
			auto m = it.get();
			auto node = nodeMap[m->objectId()];
			auto nodeData = node->flowNodeData();
			if (checkMouseInPoints(mouseWentUpPosition, nodeData, PortState::in))
			{
				auto connectionInNode = node;

				if (outPoint.portType == PortType::Out)
				{
					WtConnection connection(outPoint.portType, *connectionOutNode, outPoint.portIndex);
					WtInteraction interaction(*connectionInNode, connection, *node_scene, inPoint, outPoint, m, mOutNode);
					if (interaction.tryConnect())
					{
						sceneConnection temp;
						temp.exportNode = m;
						temp.inportNode = mOutNode;
						temp.inPoint = inPoint;
						temp.outPoint = outPoint;
						sceneConnections.push_back(temp);
						update();
					}
				}

			}
		}
	}
	drawLineFlag = false;
	update();
}

void WtNodeFlowWidget::onKeyWentDown()
{
	if (selectType > 0)
	{
		auto node = nodeMap[selectedNum];
		deleteNode(*node);
		selectType = -1;
		selectedNum = 0;
		updateAll();
	}
}

void WtNodeFlowWidget::setSelectNode(std::shared_ptr<dyno::Node> node)
{
	selectType = 2;
	selectedNum = node->objectId();
	update();
}

void WtNodeFlowWidget::paintEvent(Wt::WPaintDevice* paintDevice)
{
	Wt::WPainter painter(paintDevice);
	painter.scale(mZoomFactor, mZoomFactor);
	painter.translate(mTranslate);

	if (reorderFlag)
	{
		node_scene = new WtNodeFlowScene(&painter, mScene, selectType, selectedNum);
		node_scene->reorderAllNodes();
		reorderFlag = false;
	}

	node_scene = new WtNodeFlowScene(&painter, mScene, selectType, selectedNum);

	nodeMap = node_scene->getNodeMap();

	if (isDragging && selectType > 0 && !drawLineFlag)
	{
		auto node = nodeMap[selectedNum];
		moveNode(*node, mTranslateNode);
	}

	if (drawLineFlag)
	{
		drawSketchLine(&painter, sourcePoint, sinkPoint);
	}
}

bool WtNodeFlowWidget::checkMouseInAllRect(Wt::WPointF mousePoint)
{
	for (auto it = mScene->begin(); it != mScene->end(); it++)
	{
		auto m = it.get();
		auto node = nodeMap[m->objectId()];
		auto nodeData = node->flowNodeData();
		if (checkMouseInRect(mousePoint, nodeData))
		{
			selectedNum = m->objectId();
			canMoveNode = true;
			return true;
		}
	}
	return false;
}

bool WtNodeFlowWidget::checkMouseInHotKey0(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WRectF relativeHotkey = nodeData.getHotKey0BoundingRect();
	Wt::WPointF origin = nodeData.getNodeOrigin();

	Wt::WPointF topLeft = Wt::WPointF(relativeHotkey.topLeft().x() + origin.x(), relativeHotkey.topLeft().y() + origin.y());
	Wt::WPointF bottomRight = Wt::WPointF(relativeHotkey.bottomRight().x() + origin.x(), relativeHotkey.bottomRight().y() + origin.y());

	Wt::WRectF absRect = Wt::WRectF(topLeft, bottomRight);

	Wt::WPointF	trueMouse = Wt::WPointF(mousePoint.x() / mZoomFactor - mTranslate.x(), mousePoint.y() / mZoomFactor - mTranslate.y());

	return absRect.contains(trueMouse);
}

bool WtNodeFlowWidget::checkMouseInHotKey1(Wt::WPointF mousePoint, WtFlowNodeData nodeData)
{
	Wt::WRectF relativeHotkey = nodeData.getHotKey1BoundingRect();
	Wt::WPointF origin = nodeData.getNodeOrigin();

	Wt::WPointF topLeft = Wt::WPointF(relativeHotkey.topLeft().x() + origin.x(), relativeHotkey.topLeft().y() + origin.y());
	Wt::WPointF bottomRight = Wt::WPointF(relativeHotkey.bottomRight().x() + origin.x(), relativeHotkey.bottomRight().y() + origin.y());

	Wt::WRectF absRect = Wt::WRectF(topLeft, bottomRight);

	Wt::WPointF	trueMouse = Wt::WPointF(mousePoint.x() / mZoomFactor - mTranslate.x(), mousePoint.y() / mZoomFactor - mTranslate.y());

	return absRect.contains(trueMouse);
}

Wt::WPointF WtNodeFlowWidget::getPortPosition(Wt::WPointF origin, connectionPointData portData)
{
	if (portData.portShape == PortShape::Bullet)
	{
		Wt::WPointF topLeft = Wt::WPointF(portData.diamond_out[3].x() + origin.x(), portData.diamond_out[2].y() + origin.y());
		Wt::WPointF bottomRight = Wt::WPointF(portData.diamond_out[1].x() + origin.x(), portData.diamond_out[0].y() + origin.y());
		Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
		return Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
	}
	else if (portData.portShape == PortShape::Diamond)
	{
		Wt::WPointF topLeft = Wt::WPointF(portData.diamond[3].x() + origin.x(), portData.diamond[2].y() + origin.y());
		Wt::WPointF bottomRight = Wt::WPointF(portData.diamond[1].x() + origin.x(), portData.diamond[0].y() + origin.y());
		Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
		return Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
	}
	else if (portData.portShape == PortShape::Point)
	{
		auto rectTopLeft = portData.pointRect.topLeft();
		auto rectBottomRight = portData.pointRect.bottomRight();
		Wt::WPointF topLeft = Wt::WPointF(rectTopLeft.x() + origin.x(), rectTopLeft.y() + origin.y());
		Wt::WPointF bottomRight = Wt::WPointF(rectBottomRight.x() + origin.x(), rectBottomRight.y() + origin.y());
		Wt::WRectF diamondBoundingRect = Wt::WRectF(topLeft, bottomRight);
		return Wt::WPointF((topLeft.x() + bottomRight.x()) / 2, (topLeft.y() + bottomRight.y()) / 2);
	}
}

void WtNodeFlowWidget::deleteNode(WtNode& n)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto node = nodeData->getNode();

		for (auto c : sceneConnections)
		{
			if (c.exportNode == node || c.inportNode == node)
			{
				Wt::WMessageBox::show("Error",
					"<p>Please disconnect before deleting the node </p>",
					Wt::StandardButton::Ok);
				return;
			}
		}

		mScene->deleteNode(node);
	}
}

void WtNodeFlowWidget::disconnectionsFromNode(WtNode& node)
{

}


void WtNodeFlowWidget::moveNode(WtNode& n, const Wt::WPointF& newLocation)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr)
	{
		auto node = nodeData->getNode();
		node->setBlockCoord(newLocation.x(), newLocation.y());
	}
}

void WtNodeFlowWidget::enableRendering(WtNode& n, bool checked)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());

	if (mEditingEnabled && nodeData != nullptr) {
		auto node = nodeData->getNode();
		node->setVisible(checked);
	}
}

void WtNodeFlowWidget::enablePhysics(WtNode& n, bool checked)
{
	auto nodeData = dynamic_cast<WtNodeWidget*>(n.nodeDataModel());
	if (mEditingEnabled && nodeData != nullptr) {
		auto node = nodeData->getNode();
		node->setActive(checked);
	}
}

void WtNodeFlowWidget::disconnect(std::shared_ptr<Node> exportNode, std::shared_ptr<Node> inportNode, connectionPointData inPoint, connectionPointData outPoint, WtNode* inWtNode, WtNode* outWtNode)
{
	auto inportIndex = inPoint.portIndex;
	if (inPoint.portShape == PortShape::Diamond || inPoint.portShape == PortShape::Bullet)
	{
		exportNode->disconnect(inportNode->getImportNodes()[inportIndex]);
	}
	else if (inPoint.portShape == PortShape::Point)
	{
		auto outFieldNum = 0;
		auto outPoints = outWtNode->flowNodeData().getPointsData();
		for (auto point : outPoints)
		{
			if (point.portShape == PortShape::Point)
			{
				outFieldNum = point.portIndex;
				break;
			}
		}

		auto field = exportNode->getOutputFields()[outPoint.portIndex - outFieldNum];

		if (field != NULL)
		{
			auto node_data = inWtNode->flowNodeData();

			auto points = node_data.getPointsData();

			int fieldNum = 0;

			for (auto point : points)
			{
				if (point.portType == PortType::In)
				{
					if (point.portShape == PortShape::Bullet || point.portShape == PortShape::Diamond)
					{
						fieldNum++;
					}
				}
			}

			auto inField = inportNode->getInputFields()[inPoint.portIndex - fieldNum];

			field->disconnect(inField);
		}
	}
}