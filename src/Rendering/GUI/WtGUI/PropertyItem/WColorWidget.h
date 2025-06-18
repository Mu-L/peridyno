#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WColorPicker.h>
#include <Wt/WColor.h>

#include <WParameterDataNode.h>

#include "Field/Color.h"

class WColorWidget : public Wt::WContainerWidget
{
public:
	WColorWidget(dyno::FBase*);
	~WColorWidget();

	static Wt::WContainerWidget* WColorWidgetConstructor(dyno::FBase* field)
	{
		return new WColorWidget(field);
	};

	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField();

	Wt::Signal<int>& changeValue()
	{
		return changeValue_;
	}

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WColorPicker* mData;
	Wt::Signal<int> changeValue_;
};