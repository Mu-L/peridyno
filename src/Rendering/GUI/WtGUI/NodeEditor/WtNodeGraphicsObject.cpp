#include "WtNodeGraphicsObject.h"

WtNodePainter::WtNodePainter() {}

WtNodePainter::~WtNodePainter() {}

void WtNodePainter::paint(Wt::WPainter* painter, WtNode& node, WtFlowScene const& scene)
{
	WtNodeGeometry const& geom = node.nodeGeometry();

	WtNodeState const& state = node.nodeState();

	WtNodeGraphicsObject const& graphicsObject = node.nodeGraphicsObject();

	WtNodeDataModel const* model = node.nodeDataModel();

	drawNodeRect(painter, geom, model, graphicsObject);

	drawHotKeys(painter, geom, model, graphicsObject);

	drawConnectionPoints(painter, geom, state, model, scene, graphicsObject);

	drawModelName(painter, geom, state, model);

	drawEntryLabels(painter, geom, state, model);

	drawResizeRect(painter, geom, model);

	drawValidationRect(painter, geom, model, graphicsObject);
}

void WtNodePainter::drawNodeRect(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	auto color = nodeStyle.NormalBoundaryColor;
	if (graphicsObject.selectType() == 1)
	{
		color = nodeStyle.SelectedBoundaryColor;
	}
	if (graphicsObject.selectType() == 2)
	{
		color = nodeStyle.SelectedDragColor;
	}
	
	if (geom.hovered())
	{
		Wt::WPen p(color);
		p.setWidth(nodeStyle.HoveredPenWidth);
		painter->setPen(p);
	}
	else
	{
		Wt::WPen p(color);
		p.setWidth(nodeStyle.PenWidth);
		painter->setPen(p);
	}

	float diam = nodeStyle.ConnectionPointDiameter;
	double const radius = 6.0;

	Wt::WRectF boundary = model->captionVisible() ? Wt::WRectF(-diam, -diam, 2.0 * diam + geom.width(), 2.0 * diam + geom.height())
		: Wt::WRectF(-diam, 0.0f, 2.0 * diam + geom.width(), diam + geom.height());

	graphicsObject.setBoundingRect(boundary);

	if (model->captionVisible())
	{
		unsigned int captionHeight = geom.captionHeight();

		double captionRatio = (double)captionHeight / geom.height();

		painter->setBrush(Wt::WColor(Wt::StandardColor::White));

		painter->drawRect(boundary);

		Wt::WPen p;
		p.setColor(color);
		p.setWidth(nodeStyle.PenWidth);

		painter->setPen(p);
		painter->drawLine(Wt::WPointF(-diam, geom.captionHeight()), Wt::WPointF(diam + geom.width(), geom.captionHeight()));
	}
	else
	{
		painter->drawRect(boundary);
	}
}

void WtNodePainter::drawConnectionPoints(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeState const& state,
	WtNodeDataModel const* model,
	WtFlowScene const& scene,
	WtNodeGraphicsObject const& graphicsObject
)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	auto const& connectionStyle = WtStyleCollection::connectionStyle();

	float diameter = nodeStyle.ConnectionPointDiameter;

	auto reducedDiameter = diameter * 0.6;

	std::vector<connectionPointData> pointsData;

	for (PortType portType : {PortType::Out, PortType::In})
	{
		size_t n = state.getEntries(portType).size();

		for (unsigned int i = 0; i < n; i++)
		{
			connectionPointData pointData;
			pointData.portType = portType;

			Wt::WPointF p = geom.portScenePosition(i, portType);

			//TODO:Bug
			auto const& dataType = model->dataType(portType, i);

			bool canConnect = (state.getEntries(portType)[i].empty() ||
				(portType == PortType::Out &&
					model->portOutConnectionPolicy(i) == WtNodeDataModel::ConnectionPolicy::Many));

			double r = 1.0;

			if (state.isReacting() && canConnect && portType == state.reactingPortType())
			{
				Wt::WPointF diff = Wt::WPointF(geom.draggingPos().x() - p.x(), geom.draggingPos().y() - p.y());

				double dist = std::sqrt(diff.x() * diff.x() + diff.y() * diff.y());

				bool typeConvertable = false;

				{
					if (portType == PortType::In)
					{
						typeConvertable = scene.registry().getTypeConverter(state.reactingDataType(), dataType) != nullptr;
					}
					else
					{
						typeConvertable = scene.registry().getTypeConverter(dataType, state.reactingDataType()) != nullptr;
					}
				}

				if (state.reactingDataType().id == dataType.id || typeConvertable)
				{
					double const thres = 40.0;
					r = (dist < thres) ?
						(2.0 - dist / thres) :
						1.0;
				}
				else
				{
					double const thres = 80.0;
					r = (dist < thres) ?
						(dist / thres) :
						1.0;
				}
			}

			//TODO: change the ports appearance according to the connection states
			if (connectionStyle.useDataDefinedColors())
			{
				painter->setBrush(connectionStyle.normalColor(dataType.id));
			}
			else
			{
				painter->setBrush(nodeStyle.ConnectionPointColor);
			}

			double w = diameter * 0.4 * r;
			double h = diameter * 0.8 * r;
			Wt::WPointF vert[5];
			vert[0] = Wt::WPointF(p.x() - w, p.y() - h);
			vert[1] = Wt::WPointF(p.x() - w, p.y() + h);
			vert[2] = Wt::WPointF(p.x() + w, p.y() + h);
			vert[3] = Wt::WPointF(p.x() + (2.5 * w), p.y());
			vert[4] = Wt::WPointF(p.x() + w, p.y() - h);

			double dr = diameter * 1 * r;
			Wt::WPointF diamond[4];
			diamond[0] = Wt::WPointF(p.x(), p.y() + dr);
			diamond[1] = Wt::WPointF(p.x() + dr, p.y());
			diamond[2] = Wt::WPointF(p.x(), p.y() - dr);
			diamond[3] = Wt::WPointF(p.x() - dr, p.y());

			double odr = diameter * 1.1 * r;
			double idr = diameter * 0.3 * r;
			Wt::WPointF diamond_out[4];
			diamond_out[0] = Wt::WPointF(p.x(), p.y() + odr);
			diamond_out[1] = Wt::WPointF(p.x() + odr, p.y());
			diamond_out[2] = Wt::WPointF(p.x(), p.y() - odr);
			diamond_out[3] = Wt::WPointF(p.x() - odr, p.y());

			Wt::WPointF diamond_inner[4];
			diamond_inner[0] = Wt::WPointF(p.x(), p.y() + idr);
			diamond_inner[1] = Wt::WPointF(p.x() + idr, p.y());
			diamond_inner[2] = Wt::WPointF(p.x(), p.y() - idr);
			diamond_inner[3] = Wt::WPointF(p.x() - idr, p.y());

			double rx = reducedDiameter * r;
			double ry = reducedDiameter * r;
			Wt::WRectF drawRect = Wt::WRectF(p.x() - rx, p.y() - ry, 2 * rx, 2 * ry);

			switch (dataType.shape)
			{
			case PortShape::Point:
				painter->drawEllipse(drawRect);
				pointData.portShape = PortShape::Point;
				pointData.pointRect = drawRect;
				pointData.portIndex = (PortIndex)i;
				break;
			case PortShape::Bullet:
				painter->drawPolygon(diamond_out, 4);
				painter->setBrush(Wt::StandardColor::White);
				painter->drawPolygon(diamond_inner, 4);
				pointData.portShape = PortShape::Bullet;
				pointData.portIndex = (PortIndex)i;
				pointData.diamond_out[0] = diamond_out[0];
				pointData.diamond_out[1] = diamond_out[1];
				pointData.diamond_out[2] = diamond_out[2];
				pointData.diamond_out[3] = diamond_out[3];
				break;
			case PortShape::Diamond:
				painter->drawPolygon(diamond, 4);
				pointData.portShape = PortShape::Diamond;
				pointData.portIndex = (PortIndex)i;
				pointData.diamond[0] = diamond[0];
				pointData.diamond[1] = diamond[1];
				pointData.diamond[2] = diamond[2];
				pointData.diamond[3] = diamond[3];
				break;

			default:
				break;
			}
			pointsData.push_back(pointData);
		}
	};
	graphicsObject.setPointsData(pointsData);
}

void WtNodePainter::drawModelName(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeState const& state,
	WtNodeDataModel const* model
)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	if (!model->captionVisible())
		return;

	std::string const& name = geom.strFormat(model->caption());

	Wt::WFontMetrics metrics = painter->device()->fontMetrics();

	Wt::WPointF position(0, 0);

	Wt::WFont f = painter->font();

	f.setWeight(Wt::FontWeight::Bold);

	float diam = nodeStyle.ConnectionPointDiameter;

	Wt::WRectF boundary(position, position);

	painter->setFont(f);
	painter->setPen(Wt::WPen(nodeStyle.FontColor));
	painter->drawRect(boundary);
	painter->drawText(boundary, Wt::AlignmentFlag::Left, Wt::WString(name));

	f.setWeight(Wt::FontWeight::Normal);
	painter->setFont(f);
}

void WtNodePainter::drawHotKeys(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	//auto color = graphicsObject.selectType() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;
	auto color = nodeStyle.NormalBoundaryColor;
	if (graphicsObject.selectType() == 1)
	{
		color = nodeStyle.SelectedBoundaryColor;
	}
	if (graphicsObject.selectType() == 2)
	{
		color = nodeStyle.SelectedDragColor;
	}

	const Wt::WPen& pen = painter->pen();

	if (model->captionVisible() && model->hotkeyEnabled())
	{
		unsigned int captionHeight = geom.captionHeight();
		unsigned int keyWidth = geom.hotkeyWidth();
		unsigned int keyShift = geom.hotkeyIncline();
		unsigned int keyOffset = geom.hotkeyOffset();

		float diam = nodeStyle.ConnectionPointDiameter;

		Wt::WPen p(color);
		p.setWidth(nodeStyle.PenWidth);
		painter->setPen(p);

		if (graphicsObject.hotKey0Hovered())
		{
			Wt::WPen p1(color);
			p1.setWidth(nodeStyle.HoveredPenWidth);
			painter->setPen(p1);
		}
		else
		{
			Wt::WPen p1(color);
			p1.setWidth(nodeStyle.PenWidth);
			painter->setPen(p1);
		}

		if (graphicsObject.isHotKey0Checked())
		{
			painter->setBrush(nodeStyle.GradientColor0);
		}
		else
		{
			painter->setBrush(nodeStyle.HotKeyColor0);
		}

		Wt::WPointF points[4];
		points[0] = Wt::WPointF(geom.width() + diam - keyWidth - keyOffset, -diam);
		points[1] = Wt::WPointF(geom.width() + diam - keyOffset, -diam);
		//points[2] = Wt::WPointF(geom.width() + diam - keyShift - keyOffset, captionHeight);
		//points[3] = Wt::WPointF(geom.width() + diam - keyWidth - keyShift - keyOffset, captionHeight);
		points[2] = Wt::WPointF(geom.width() + diam - keyOffset, captionHeight);
		points[3] = Wt::WPointF(geom.width() + diam - keyWidth - keyOffset, captionHeight);

		//painter->drawPolygon(points, 4);

		Wt::WRectF hotKey0Rect = Wt::WRectF(points[0], points[2]);

		painter->drawRect(hotKey0Rect);

		graphicsObject.setHotKey0BoundingRect(hotKey0Rect);

		if (graphicsObject.hotKey1Hovered())
		{
			Wt::WPen p2(color);
			p2.setWidth(nodeStyle.HoveredPenWidth);
			painter->setPen(p2);
		}
		else
		{
			Wt::WPen p2(color);
			p2.setWidth(nodeStyle.PenWidth);
			painter->setPen(p2);
		}

		if (graphicsObject.isHotKey1Checked())
		{
			painter->setBrush(nodeStyle.GradientColor0);
		}
		else
		{
			painter->setBrush(nodeStyle.HotKeyColor1);
		}

		points[0] = Wt::WPointF(geom.width() + diam - keyWidth - keyOffset, -diam);
		//points[1] = Wt::WPointF(geom.width() + diam - keyWidth - keyShift - keyOffset, captionHeight);
		//points[2] = Wt::WPointF(geom.width() + diam - keyWidth - keyShift - keyWidth - keyOffset, captionHeight);
		points[3] = Wt::WPointF(geom.width() + diam - keyWidth - keyWidth - keyOffset, -diam);

		points[1] = Wt::WPointF(geom.width() + diam - keyWidth - keyOffset, captionHeight);
		points[2] = Wt::WPointF(geom.width() + diam - keyWidth - keyWidth - keyOffset, captionHeight);

		Wt::WRectF hotKey1Rect = Wt::WRectF(points[3], points[1]);

		painter->drawRect(hotKey1Rect);

		graphicsObject.setHotKey1BoundingRect(hotKey1Rect);

		//painter->drawPolygon(points, 4);
	}
}

void WtNodePainter::drawEntryLabels(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeState const& state,
	WtNodeDataModel const* model)
{
	for (PortType portType : { PortType::Out, PortType::In })
	{
		auto const& nodeStyle = model->nodeStyle();

		auto& entries = state.getEntries(portType);

		size_t n = entries.size();

		for (size_t i = 0; i < n; ++i)
		{
			Wt::WPointF p = geom.portScenePosition((PortIndex)i, portType);

			if (entries[i].empty())
				painter->setPen(nodeStyle.FontColorFaded);
			else
				painter->setPen(nodeStyle.FontColor);

			std::string s;

			if (model->portCaptionVisible(portType, (PortIndex)i))
				s = model->portCaption(portType, (PortIndex)i);
			else
				s = model->dataType(portType, (PortIndex)i).name;

			Wt::WFontMetrics metrics = painter->device()->fontMetrics();

			Wt::WPointF topLeft, bottomRight;

			switch (portType)
			{
			case PortType::In:
				topLeft = Wt::WPointF(p.x() + 10, p.y() - metrics.height() / 3);
				bottomRight = Wt::WPointF(p.x() + 50, p.y() + 6);
				painter->drawText(Wt::WRectF(topLeft, bottomRight), Wt::AlignmentFlag::Left, Wt::WString(s));
				break;

			case PortType::Out:
				topLeft = Wt::WPointF(p.x() - 50, p.y() - 6);
				bottomRight = Wt::WPointF(p.x() - 10, p.y() + metrics.height() / 3);
				painter->drawText(Wt::WRectF(topLeft, bottomRight), Wt::AlignmentFlag::Right, Wt::WString(s));
				break;

			default:
				break;
			}
		}
	}
}

void WtNodePainter::drawResizeRect(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model
)
{
	if (model->resizable())
	{
		painter->setBrush(Wt::StandardColor::Gray);
		painter->drawEllipse(geom.resizeRect());
	}
}

void WtNodePainter::drawValidationRect(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject
)
{
	auto modelValidationState = model->validationState();

	if (modelValidationState != NodeValidationState::Valid)
	{
		WtNodeStyle const& nodeStyle = model->nodeStyle();

		//auto color = graphicsObject.selectType() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;
		auto color = nodeStyle.NormalBoundaryColor;
		if (graphicsObject.selectType() == 1)
		{
			color = nodeStyle.SelectedBoundaryColor;
		}
		if (graphicsObject.selectType() == 2)
		{
			color = nodeStyle.SelectedDragColor;
		}

		if (geom.hovered())
		{
			Wt::WPen p(color);
			p.setWidth(Wt::WLength(nodeStyle.HoveredPenWidth, Wt::LengthUnit::Pixel));
			painter->setPen(p);
		}
		else
		{
			Wt::WPen p(color);
			p.setWidth(Wt::WLength(nodeStyle.PenWidth, Wt::LengthUnit::Pixel));
			painter->setPen(p);
		}

		if (modelValidationState == NodeValidationState::Error)
		{
			painter->setBrush(nodeStyle.ErrorColor);
		}
		else
		{
			painter->setBrush(nodeStyle.WarningColor);
		}

		double const radius = 3.0;

		float diam = nodeStyle.ConnectionPointDiameter;

		Wt::WRectF boundary(-diam,
			-diam + geom.height() - geom.validationHeight(),
			2.0 * diam + geom.width(),
			2.0 * diam + geom.validationHeight());

		painter->drawRect(boundary);

		painter->setBrush(Wt::StandardColor::Gray);

		//No ErrorMsg
	}
}

//WtNodeGraphicsObject

WtNodeGraphicsObject::WtNodeGraphicsObject(WtFlowScene& scene, WtNode& node, Wt::WPainter* painter, int selectType)
	: _scene(scene)
	, _node(node)
	, _painter(painter)
	, _locked(false)
	, _flowNodeData(node.flowNodeData())
	, _selectType(selectType)
{
}

WtNodeGraphicsObject::~WtNodeGraphicsObject() {}

WtNode& WtNodeGraphicsObject::node()
{
	return _node;
}

WtNode const& WtNodeGraphicsObject::node() const
{
	return _node;
}

void WtNodeGraphicsObject::embedQWidget()
{
}

Wt::WRectF WtNodeGraphicsObject::boundingRect() const
{
	return _node.nodeGeometry().boundingRect();
}

void WtNodeGraphicsObject::setGeometryChanged()
{
	//prepareGeometryChange();
}

void WtNodeGraphicsObject::moveConnections() const
{
	WtNodeState const& nodeState = _node.nodeState();

	//for (PortType portType : {PortType::In, PortType::Out})
	//{
	//	auto const& connectionEntries = nodeState.getEntries(portType);

	//	for (auto const& connections : connectionEntries)
	//	{
	//		for (auto& con : connections)
	//			con.second->getConnectionGraphicsObject().move();
	//	}
	//}
}

void WtNodeGraphicsObject::lock(bool locked)
{
	_locked = locked;

	//setFlag(QGraphicsItem::ItemIsMovable, !locked);
	//setFlag(QGraphicsItem::ItemIsFocusable, !locked);
	//setFlag(QGraphicsItem::ItemIsSelectable, !locked);
}

void WtNodeGraphicsObject::paint(Wt::WPainter* painter)
{
	WtNodePainter::paint(painter, _node, _scene);
}

Wt::WPointF WtNodeGraphicsObject::getPos() const
{
	return _origin;
}

void WtNodeGraphicsObject::setPos(int x, int y)
{
	_origin.setX(x);
	_origin.setY(y);
	_painter->translate(_origin);
	_flowNodeData.setNodeOrigin(_origin);
	paint(_painter);
	_origin.setX(-x);
	_origin.setY(-y);
	_painter->translate(_origin);
}

void WtNodeGraphicsObject::setPos(Wt::WPointF pos)
{
	_origin = pos;
	_painter->translate(_origin);
	_flowNodeData.setNodeOrigin(_origin);
	paint(_painter);
	_origin.setX(-pos.x());
	_origin.setY(-pos.y());
	_painter->translate(_origin);
}