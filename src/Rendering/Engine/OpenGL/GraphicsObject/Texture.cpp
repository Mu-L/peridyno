#include "Texture.h"
#include <glad/glad.h>
#include <iostream>

namespace dyno
{
	Texture::Texture()
	{
		// default value...
		this->format = GL_RGBA;
		this->internalFormat = GL_RGBA32F;
		this->type = GL_FLOAT;
	}

	void Texture::create()
	{
		if (target == -1) {
			std::cerr << "Failed to create texture, wrong target id: " << target << std::endl;
			return;
		}
		glGenTextures(1, &id);
		glCheckError();
	}

	void Texture::release()
	{
		glDeleteTextures(1, &id);
		// reset object id
		id = GL_INVALID_INDEX;
	}

	void Texture::bind()
	{
		glBindTexture(target, id);
	}

	void Texture::bind(int slot)
	{
		glActiveTexture(slot);
		glBindTexture(target, id);
		glCheckError();
	}

	void Texture::unbind()
	{
		glBindTexture(target, 0);
		glCheckError();
	}

	void Texture::clear(void* value)
	{
		glClearTexImage(id, 0, format, type, value);
		glCheckError();
	}

	Texture2D::Texture2D()
	{
		this->target = GL_TEXTURE_2D;
		this->minFilter = GL_LINEAR;	//GL_LINEAR_MIPMAP_LINEAR
		this->maxFilter = GL_LINEAR;
	}

	void Texture2D::create()
	{
		Texture::create();

		glBindTexture(target, id);
		glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
		glTexParameteri(target, GL_TEXTURE_MAG_FILTER, maxFilter);
		glCheckError();
	}

	void Texture2D::dump(void* pixels)
	{
		glBindTexture(target, id);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glGetTexImage(target, 0, internalFormat, type, pixels);
		glCheckError();
	}

	void Texture2D::resize(int w, int h)
	{
		glBindTexture(target, id);
		glTexImage2D(target, 0, internalFormat, w, h, 0, format, type, 0);
		glCheckError();
	}


	void Texture2D::load(int w, int h, void* data)
	{
		glBindTexture(target, id);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(target, 0, internalFormat, w, h, 0, format, type, data);
		glCheckError();
	}

	void Texture2D::genMipmap()
	{
		glBindTexture(target, id);
		glGenerateMipmap(target);
		glCheckError();
	}

	Texture2DMultiSample::Texture2DMultiSample()
	{
		this->target = GL_TEXTURE_2D_MULTISAMPLE;
	}

	void Texture2DMultiSample::resize(int w, int h, unsigned int nSamples)
	{
		this->samples = nSamples;

		glBindTexture(target, id);
		glTexImage2DMultisample(target, samples, internalFormat, w, h, GL_TRUE);
		glCheckError();
	}

	TextureCube::TextureCube()
	{
		this->target = GL_TEXTURE_CUBE_MAP;
	}

}

