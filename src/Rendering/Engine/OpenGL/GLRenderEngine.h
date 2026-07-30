/**
 * Copyright 2017-2021 Jian SHI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <vector>

#include <RenderEngine.h>

#include "GraphicsObject/Buffer.h"
#include "GraphicsObject/Texture.h"
#include "GraphicsObject/Framebuffer.h"
#include "GraphicsObject/Shader.h"
#include "GraphicsObject/Mesh.h"


namespace dyno
{
	class SSAO;
	class FXAA;
	class Envmap;
	class ShadowMap;
	class GLRenderHelper;
	class GLVisualModule;
	class SceneGraph;

	class GLRenderEngine : public RenderEngine
	{
	public:
		GLRenderEngine();
		~GLRenderEngine();
			   
		virtual void initialize() override;
		virtual void terminate() override;

		virtual void draw(dyno::SceneGraph* scene, const RenderParams& rparams, const Vec2i p = Vec2i(0)) override;

		virtual std::string name() const override;

		// get the selected nodes on given rect area
		Selection select(int x, int y, int w, int h) override;

		// use MSAA samples
		void setMSAA(int samples);
		int  getMSAA() const;

		void setFXAA(bool flag);
		int getFXAA() const;

		void setShadowMapSize(int size);
		int  getShadowMapSize() const;

		void setShadowBlurIters(int iters);
		int  getShadowBlurIters() const;

		void setDefaultEnvmap() override;
		void setEnvmap(const std::string& path);

		void setEnvStyle(EEnvStyle style) override;

		inline std::string getEnvmapFilePath() { return mEnvmapFilePath; }

		int  getShadowMapSize();
		void updateShadowMapAttribute()override;
	protected:
		void createFramebuffer();
		void resizeFramebuffer(int w, int h, int samples);
		void setupTransparencyPass();
		void updateRenderItems(dyno::SceneGraph* scene);

	private:

		// objects to render
		struct RenderItem {
			std::shared_ptr<Node>			node;
			std::shared_ptr<GLVisualModule> visualModule;

			bool operator==(const RenderItem& item) {
				return node == item.node && visualModule == item.visualModule;
			}
		};

		std::vector<RenderItem> mRenderItems;

	protected:

		//Texture2DMultiSample	mColorCorrectTex;
		Program* mPostProcessProgram;

		// internal framebuffer
		Framebuffer				mFramebuffer;
		Texture2DMultiSample	mColorTex;
		Texture2DMultiSample	mDepthTex;
		Texture2DMultiSample	mIndexTex;			// indices for object/mesh/primitive etc.

		// non-multisample framebuffer for select
		Framebuffer				mSelectFramebuffer;
		Texture2D				mSelectIndexTex;

		// for Weighted Blended OIT (WBOIT)
		// dual G-buffers, kept multisample to match the opaque MSAA pass
		Texture2DMultiSample	mAccumTex;			// RGBA16F: premultiplied, weighted color
		Texture2DMultiSample	mRevealTex;		// RGBA16F: revealage (product of 1 - alpha)
		Texture2D				mAccumResolveTex;	// single-sample resolve target for composite
		Texture2D				mRevealResolveTex;	// single-sample resolve target for composite
		Framebuffer				mWBOITFramebuffer;	// accum(0) + reveal(2) sharing the opaque depth
		Framebuffer				mWBOITResolveFBO;	// single-sample accum(0) + reveal(2)
		Program*				mWBOITCompositeProgram;

		GLRenderHelper*			mRenderHelper;
		ShadowMap*				mShadowMap = NULL;

		// anti-aliasing
		
		// MSAA samples
		int						mMSAASamples = 4;

		// FXAA
		bool					bEnableFXAA = false;
		FXAA*					mFXAAFilter;

		//ShadowType
		int						mShadowType = 2;

		// Envmap
		std::string				mEnvmapFilePath = getAssetPath() + "textures/hdr/venice_dawn_1_4k.hdr";
		Envmap*					mEnvmap = NULL;
		
		Mesh* mScreenQuad = 0;
	};
};
