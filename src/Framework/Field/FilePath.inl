#ifndef FILEPATH_SERIALIZATION
#define FILEPATH_SERIALIZATION

#include "Field.h"

namespace dyno {
	template<>
	inline std::string FVar<FilePath>::serialize()
	{
		if (isEmpty())
			return "";

		FilePath val = this->getValue();

		return val.string();
	}

	template<>
	inline bool FVar<FilePath>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		this->setValue(str);

		return true;
	}

	template class FVar<FilePath>;
}

#endif