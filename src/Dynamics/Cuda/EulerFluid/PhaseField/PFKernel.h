#pragma once

#include "Platform.h"
#include "Vector.h"
#include "Array/Array.h"
#include "Array/Array3D.h"

//#define INPUT_BRUSH

namespace dyno{

#define RHO1 1000.0f
#define RHO2 10.0f

#define VIS1 10000.0f
#define VIS2 10000.0f

#define MU_INF 100.0f
#define MU_0   0.01f
#define ALPHA  1.0f
#define SCALING 100.0f
#define EXP_N  0.4f

#define MASS_TRESHOLD 0.005f
#define MASS_THESHOLD2 0.5f

#define LAUNCH_KERNEL(name, grid, block, ...)																				\
{																															\
	dim3 gridDims, blockDims;																								\
	computeGridSize3D(grid, block, gridDims, blockDims);																	\
	##name << < gridDims, blockDims>> >(__VA_ARGS__);																		\
	cudaError_t code = cudaDeviceSynchronize();																				\
	if (code != cudaSuccess)																								\
	{																														\
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); exit(code);					\
	}																														\
}

// 	static uint iDivUp(uint a, uint b)
// 	{
// 		return (a % b != 0) ? (a / b + 1) : (a / b);
// 	}

	// compute grid and thread block size for a given number of elements
	static void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
	{
		numThreads = blockSize < n ? blockSize : n;
		numBlocks = iDivUp(n, numThreads);
	}

	static void computeGridSize2D(uint2 dims, uint2 blockSize, dim3& gridDim, dim3& blockDim)
	{
		gridDim.x = iDivUp(dims.x, blockSize.x);
		gridDim.y = iDivUp(dims.y, blockSize.y);
		gridDim.z = 1;

		blockDim.x = blockSize.x;
		blockDim.y = blockSize.y;
		blockDim.z = 1;
	}

	static void computeGridSize3D(uint3 dims, uint3 blockSize, dim3& gridDim, dim3& blockDim)
	{
		gridDim.x = iDivUp(dims.x, blockSize.x);
		gridDim.y = iDivUp(dims.y, blockSize.y);
		gridDim.z = iDivUp(dims.z, blockSize.z);

		blockDim.x = blockSize.x;
		blockDim.y = blockSize.y;
		blockDim.z = blockSize.z;
	}

	struct Coef
	{
		float a;
		float x0;
		float x1;
		float y0;
		float y1;
		float z0;
		float z1;
	};

	typedef DArray3D<int> Grid1i;
	typedef DArray3D<Vec3f> Grid3f;
	typedef DArray3D<Vec4f> Grid4f;
	typedef DArray3D<Coef> GridCoef;

	class PFKernel
	{
	public:
		static void InterpolateVelocity(Grid3f vel, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w);

		static void InterpolateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid3f vel);

		static void AdvectBackward(Grid1f dst, Grid1f src, Grid3f vel, float dt);

		//static void AdvectBackwardFromCanvas(Grid1f, Grid3f vel, float dt, uint3 simOrigin);

		static void AdvectStaggeredBackward(Grid1f dst, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float dt, int axis);

		static void AdvectForward(Grid1f dst, Grid1f src, Grid3f vel, float dt);

		//static void AdvectPigments(Grid3f pos, Grid1f mass, Grid3f vel, float dt);

		//static void DepositParticles(Grid3f gPos, Grid3f prePos, Grid4f gColor, Grid4f preColor, Grid1i gNum, Grid1i preNum, Grid1f gAcc, Grid1f preAcc, Grid1u gMutex, Grid1f mass);

		//static void SeedNewParticles(Grid3f gPos, Grid4f gColor, Grid4f preColor, Grid1i gNum, Grid1f gAcc, Grid1f preAcc, Grid1f mass, Grid3f vel, float dt);

		//static void RasterizePigments2Grid(Grid4f pgOnCenter, Grid3f pos, Grid4f color, Grid1i counter, uint3 winOrigin);

		static void AdvectBackward(Grid3f dst, Grid3f src, Grid3f vel, float dt);

		//Similar to AdvectBackward, except special treatment near boundary
		static void AdvectBackwardColor(Grid4f dst, Grid4f src, Grid3f vel, float dt);

		static void AdvectBFECC(Grid4f dst, Grid4f src, Grid4f buf, Grid1f weight, Grid3f vel, float dt);

		static void AdvectForward(Grid4f dst, Grid4f src, Grid3f vel, Grid1f weight, float dt);

// 		static void ExtraploateColor(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, int iteration);

//		static void ExtraploateMass(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, float* density_brush, int3 simOrigin, int3 brushOrigin, int iteration);

// 		static void ExtraploateColor(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, Grid3f pos, Grid4f color, Grid1i num, Grid1f accuracy, int iteration);

		static void Sharpening(Grid1f dst, Grid3f dir, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f omega, float gamma,
			float h, float dt);

		static void Jacobi(Grid1f dst, Grid1f src, Grid1f buf, Grid3f vel, float a, float c, int iteration);

		static void ApplyViscosity(Grid3f dst, Grid3f src, Grid3f buf, Grid1f mass, float a, int iteration);

		static void PrepareForProjection(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, GridCoef coefMatrix, Grid1f RHS, Grid1f mass, float h, float dt);

		static void Projection(Grid1f pressure, Grid1f buf, GridCoef coefMatrix, Grid1f RHS, int numIter);

		static void UpdateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f pressure, Grid1f mass, float h, float dt);

		static void ApplyDragForce(Grid3f vel, float dt);

		static void ApplyGravity(Grid1f u, Grid1f v, Grid1f w, Vec3f g, 
			int nx,
			int ny,
			int nz, float dt);

		static void SetU(Grid1f vel_u);

		static void SetV(Grid1f vel_v);

		static void SetW(Grid1f vel_w);

		static float calcualteTotalMass(Grid1f mass);

		//static void InputDensity(Grid1f mass, Grid1f mb0, Grid1f mb1, Grid4f pigments, Grid4f cb0, Grid4f cb1, Grid4f subPigs, uint3 cvOrigin, float dt);

		//static void InputVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f mass, uint3 cvOrigin, float dt);

		//static void SetupDensityField(Grid1f field, Grid4f pigment, uint3 cvOrigin);

		//static void SetupParticles(Grid3f gPos, Grid4f gColor);

		//static void InitialCanvas();

		//static void MoveSimWindow(Grid1f mass, Grid1f pre_mass, Grid4f pigments, Grid4f pre_pigments, Grid1f vel_u, Grid1f pre_vel_u, Grid1f vel_v, Grid1f pre_vel_v, Grid1f vel_w, Grid1f pre_vel_w, uint3 originLast, uint3 originNow);

		//static void MoveSimWindow(Grid1f mass, Grid1f pre_mass, Grid1f vel_u, Grid1f pre_vel_u, Grid1f vel_v, Grid1f pre_vel_v, Grid1f vel_w, Grid1f pre_vel_w, uint3 originLast, uint3 originNow);

		//static void MovePigments(Grid4f color, Grid4f preColor, Grid3f pos, Grid3f prePos, Grid1i num, Grid1i preNum, Grid1f acc, Grid1f preAcc, uint3 originLast, uint3 originNow);

		//static void Step();

		//static void SetPhaseFieldParameters(PFParameter* params);
	};
}