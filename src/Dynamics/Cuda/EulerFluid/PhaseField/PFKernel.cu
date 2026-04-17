#include "PFKernel.h"
// #include "cuda/helper_math.h"
// #include "../../common/cudaEssential.h"
// #include "../exchange_define.cuh"

#include "Algorithm/Reduction.h"


namespace dyno{

		//__constant__ PFParameter pfParams;

		template<typename T>
		__global__ void K_CopyData(T dst, T src)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			if (i >= src.nx()) return;
			if (j >= src.ny()) return;
			if (k >= src.nz()) return;

			dst(i, j, k) = src(i, j, k);
		}

		__global__ void K_SetVelocityBoundary(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = vel_v.nx();
			int ny = vel_w.ny();
			int nz = vel_u.nz();

			if (i >= vel_v.nx()) return;
			if (j >= vel_w.ny()) return;
			if (k >= vel_u.nz()) return;

			if (k == nz - 1) { vel_u(i, j, k) = vel_u(i, j, k - 1); vel_w(i, j, k) = vel_w(i, j, k - 1); return; }
		}

		__global__ void K_InterpolateVelocity(Grid3f vel, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			if (i >= vel.nx()) return;
			if (j >= vel.ny()) return;
			if (k >= vel.nz()) return;

			float a = 1.0f;

			Vec3f vel_ijk;
			vel_ijk.x = 0.5f*(vel_u(i, j, k) + vel_u(i + 1, j, k));
			vel_ijk.y = 0.5f*(vel_v(i, j, k) + vel_v(i, j + 1, k));
			vel_ijk.z = 0.5f*(vel_w(i, j, k) + vel_w(i, j, k + 1));

// 			if (i == 10)
// 			{
// 				printf("%f ", 0.5f*(vel_w(i, j, k) + vel_w(i, j, k + 1)));
// 			}

			vel(i, j, k) = vel_ijk;
		}

		void PFKernel::InterpolateVelocity(Grid3f vel, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(vel.nx(), vel.ny(), vel.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

//			K_SetVelocityBoundary << < gridDims, blockDims >> >(vel_u, vel_v, vel_w);
			K_InterpolateVelocity << < gridDims, blockDims >> >(vel, vel_u, vel_v, vel_w);
		}

		__global__ void K_InterpolateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid3f vel)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			if (i >= vel.nx() - 1) return;
			if (j >= vel.ny() - 1) return;
			if (k >= vel.nz() - 1) return;

// 			if (i == vel.nx - 1) { vel_u(0, j, k) = 0.0f;  vel_u(vel.nx, j, k) = 0.0f; return; }
// 			if (j == vel.ny - 1) { vel_v(i, 0, k) = 0.0f;  vel_v(i, vel.ny, k) = 0.0f; return; }
// 			if (k == vel.nz - 1) { vel_w(i, j, 0) = 0.0f;  vel_w(i, j, vel.nz) = 0.0f; return; }

			float vx, vy, vz;

			auto v = vel(i, j, k);
			auto v_i_plus = vel(i + 1, j, k);
			auto v_j_plus = vel(i, j + 1, k);
			auto v_k_plus = vel(i, j, k + 1);

			vx = 0.5f*(v.x + v_i_plus.x);
			vy = 0.5f*(v.y + v_j_plus.y);
			vz = 0.5f*(v.z + v_k_plus.z);

			if(i < vel.nx() - 1)vel_u(i + 1, j, k) = vx;
			if(j < vel.ny() - 1)vel_v(i, j + 1, k) = vy;
			if(k < vel.nz() - 1)vel_w(i, j, k + 1) = vz;
		}

		__global__ void K_DampVelocity(Grid3f vel, float radius)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			if (i >= vel.nx() - 1) return;
			if (j >= vel.ny() - 1) return;
			if (k >= vel.nz() - 1) return;


			float w = 1.0f;

			float d = sqrt(((float)i - vel.nx() / 2.0f)*((float)i - vel.nx() / 2.0f) + ((float)j - vel.ny() / 2.0f)*((float)j - vel.ny() / 2.0f));
			if (d > 1.5f*radius)
			{
				w = 0.0f;
			}
			else if (d > 0.5f*radius)
			{
				w = 1.0f - (d - 0.5f*radius) / radius;
			}

			vel(i, j, k) = w*vel(i, j, k);
		}

		void PFKernel::InterpolateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid3f vel)
		{
			uint3 gDIms = make_uint3(vel.nx(), vel.ny(), vel.nz());
			uint3 bDims = make_uint3(32, 8, 1);

			vel_u.reset();
			vel_v.reset();
			vel_w.reset();

			//LAUNCH_KERNEL(K_DampVelocity, gDIms, bDims, vel, radius);
			LAUNCH_KERNEL(K_InterpolateVelocity, gDIms, bDims, vel_u, vel_v, vel_w, vel);
		}

		__global__ void K_AdvectBackward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			auto vel_ijk = vel(i, j, k);
			
			int ix, iy, iz;
			float fx, fy, fz;
			float w000, w100, w010, w001, w111, w011, w101, w110;

			fx = i - vel_ijk.x*dt;
			fy = j - vel_ijk.y*dt;
			fz = k - vel_ijk.z*dt;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > nx - 1) fx = nx - 1.0f;
			if (fy < 0.0f) fy = 0.0f;
			if (fy > ny - 1) fy = ny - 1.0f;
			if (fz < 0.0f) fz = 0.0f;
			if (fz > nz - 1) fz = nz - 1.0f;

			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
			fx -= ix;			fy -= iy;			fz -= iz;

			if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
			if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
			if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx*(1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy*(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx*fy*fz;
			w011 = (1.0f - fx)*fy*fz;
			w101 = fx*(1.0f - fy)*fz;
			w110 = fx*fy*(1.0f - fz);

			int nxy = nx*ny;
			int k0 = ix + iy*nx + iz*nxy;

			dst(i, j, k) = w000*src[k0] + w100*src[k0 + 1] + w010*src[k0 + nx] + w001*src[k0 + nxy]
				+ w111*src[k0 + 1 + nx + nxy] + w011*src[k0 + nx + nxy] + w101*src[k0 + 1 + nxy] + w110*src[k0 + 1 + nx];
		}

		void PFKernel::AdvectBackward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_AdvectBackward <<< gridDims, blockDims >>>(dst, src, vel, dt);
		}

// 		__global__ void K_AdvectBackwardFromCanvas(Grid1f dst, cudaSurfaceObject_t surf_paintdensity, Grid3f vel, float dt, uint3 simOrigin, uint3 cvDims)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = dst.nx;
// 			int ny = dst.ny;
// 			int nz = dst.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			float3 vel_ijk = vel(i, j, k);
// 
// 			int ix, iy, iz;
// 			float fx, fy, fz;
// 			float w000, w100, w010, w001, w111, w011, w101, w110;
// 
// 			int i_cv = i + simOrigin.x;
// 			int j_cv = j + simOrigin.y;
// 			int k_cv = k + simOrigin.z;
// 
// 			fx = i_cv - vel_ijk.x*dt;
// 			fy = j_cv - vel_ijk.y*dt;
// 			fz = k_cv - vel_ijk.z*dt;
// 
// 			if (fx < 0.0f) fx = 0.0f;
// 			if (fx > cvDims.x - 1) fx = cvDims.x - 1.0f;
// 			if (fy < 0.0f) fy = 0.0f;
// 			if (fy > cvDims.y - 1) fy = cvDims.y - 1.0f;
// 			if (fz < 0.0f) fz = 0.0f;
// 			if (fz > cvDims.z - 1) fz = cvDims.z - 1.0f;
// 
// 			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
// 			fx -= ix;			fy -= iy;			fz -= iz;
// 
// 			if (ix == cvDims.x - 1) { ix = cvDims.x - 2; fx = 1.0f; }
// 			if (iy == cvDims.y - 1) { iy = cvDims.y - 2; fy = 1.0f; }
// 			if (iz == cvDims.z - 1) { iz = cvDims.z - 2; fz = 1.0f; }
// 
// 			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
// 			w100 = fx*(1.0f - fy)*(1.0f - fz);
// 			w010 = (1.0f - fx)*fy*(1.0f - fz);
// 			w001 = (1.0f - fx)*(1.0f - fy)*fz;
// 			w111 = fx*fy*fz;
// 			w011 = (1.0f - fx)*fy*fz;
// 			w101 = fx*(1.0f - fy)*fz;
// 			w110 = fx*fy*(1.0f - fz);
// 
// 			float d000, d100, d010, d001, d111, d011, d101, d110;
// 			surf3Dread(&d000, surf_paintdensity, ix*sizeof(float), iy, iz);
// 			surf3Dread(&d100, surf_paintdensity, (ix+1)*sizeof(float), iy, iz);
// 			surf3Dread(&d010, surf_paintdensity, ix*sizeof(float), iy+1, iz);
// 			surf3Dread(&d001, surf_paintdensity, ix*sizeof(float), iy, iz+1);
// 			surf3Dread(&d111, surf_paintdensity, (ix+1)*sizeof(float), iy+1, iz+1);
// 			surf3Dread(&d011, surf_paintdensity, ix*sizeof(float), iy+1, iz+1);
// 			surf3Dread(&d101, surf_paintdensity, (ix+1)*sizeof(float), iy, iz+1);
// 			surf3Dread(&d110, surf_paintdensity, (ix+1)*sizeof(float), iy+1, iz);
// 			dst(i, j, k) = w000*d000 + w100*d100 + w010*d010 + w001*d001
// 				+ w111*d111 + w011*d011 + w101*d101 + w110*d110;
// 		}

// 		void PFKernel::AdvectBackwardFromCanvas(Grid1f dst, Grid3f vel, float dt, uint3 simOrigin)
// 		{
// 			cudaSurfaceObject_t PaintDensity_SurfaceObj = 0; 
// 			PaintDensity_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.PaintDensity_cudaArray);
// 
// 			uint3 cvSize = PARAM_COMMON.CanvasDims;
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(dst.nx, dst.ny, dst.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 			K_AdvectBackwardFromCanvas << <gridDims, blockDims >> >(dst, PaintDensity_SurfaceObj, vel, dt, simOrigin, cvSize);
// 		}

		__global__ void K_AdvectBackwardColor(Grid4f dst, Grid4f src, Grid3f vel, float dt)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			Vec3f vel_ijk = vel(i, j, k);

			int ix, iy, iz;
			float fx, fy, fz;
			float w000, w100, w010, w001, w111, w011, w101, w110;

			fx = i - vel_ijk.x*dt;
			fy = j - vel_ijk.y*dt;
			fz = k - vel_ijk.z*dt;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > nx - 1) fx = nx - 1.0f;
			if (fy < 0.0f) fy = 0.0f;
			if (fy > ny - 1) fy = ny - 1.0f;
			if (fz < 0.0f) fz = 0.0f;
			if (fz > nz - 1) fz = nz - 1.0f;

			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
			fx -= ix;			fy -= iy;			fz -= iz;

			if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
			if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
			if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx*(1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy*(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx*fy*fz;
			w011 = (1.0f - fx)*fy*fz;
			w101 = fx*(1.0f - fy)*fz;
			w110 = fx*fy*(1.0f - fz);

			int nxy = nx*ny;
			int k0 = ix + iy*nx + iz*nxy;

			float alpha = 0.0f;
			float total_weight = 0.0f;
			Vec4f c_t;
			Vec4f c_sum = Vec4f(0.0f);

			c_t = src[k0];
			alpha = w000*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + 1];
			alpha = w100*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + nx];
			alpha = w010*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + nxy];
			alpha = w001*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + 1 + nx + nxy];
			alpha = w111*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + nx + nxy];
			alpha = w011*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + 1 + nxy];
			alpha = w101*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			c_t = src[k0 + 1 + nx];
			alpha = w110*c_t.w;	total_weight += alpha;	c_sum += c_t*alpha;

			if (total_weight > EPSILON)
			{
				c_sum /= total_weight;
			}
			else
			{
				c_sum = Vec4f(0.0f);
			}

			dst(i, j, k) = c_sum;// w000*src[k0] + w100*src[k0 + 1] + w010*src[k0 + nx] + w001*src[k0 + nxy]
			//	+ w111*src[k0 + 1 + nx + nxy] + w011*src[k0 + nx + nxy] + w101*src[k0 + 1 + nxy] + w110*src[k0 + 1 + nx];
		}

		void PFKernel::AdvectBackwardColor(Grid4f dst, Grid4f src, Grid3f vel, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_AdvectBackwardColor << < gridDims, blockDims >> >(dst, src, vel, dt);
		}


		__global__ void K_UpdateBFECC(Grid4f dst, Grid4f src)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int id = dst.index(i, j, k);
			auto phi_n = dst[id];
			auto phi_s = src[id];

			auto phi_new = phi_n + 0.5f*(phi_n-phi_s);

			glm::clamp(phi_new.x, 0.0f, 1.0f);
			glm::clamp(phi_new.y, 0.0f, 1.0f);
			glm::clamp(phi_new.z, 0.0f, 1.0f);
			glm::clamp(phi_new.w, 0.0f, 1.0f);
			dst[id] = phi_new;
		}

		__global__ void K_Combine1stAnd2nd(Grid4f secondOrder, Grid4f firstOrder, float alpha)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = secondOrder.nx();
			int ny = secondOrder.ny();
			int nz = secondOrder.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int id = secondOrder.index(i, j, k);

			secondOrder[id] = alpha*secondOrder[id] + (1.0f - alpha)*firstOrder[id];
		}

		__global__ void K_SmoothingBFECC(Grid4f dst, Grid4f src, float alpha)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int k0 = dst.index(i, j, k);

			int i_minus = glm::clamp(i - 1, 0, nx - 1);
			int i_plus = glm::clamp(i + 1, 0, nx - 1);

			int j_minus = glm::clamp(j - 1, 0, ny - 1);
			int j_plus = glm::clamp(j + 1, 0, ny - 1);

			int k_minus = glm::clamp(k - 1, 0, nz - 1);
			int k_plus = glm::clamp(k + 1, 0, nz - 1);


			dst[k0] = (alpha*src[k0] + (1.0f-alpha)/6.0f*(src(i_plus, j, k) + src(i_minus, j, k) + src(i, j_plus, k) + src(i, j_minus, k) + src(i, j, k_plus) + src(i, j, k_minus)));
		}

		void PFKernel::AdvectBFECC(Grid4f dst, Grid4f src, Grid4f buf, Grid1f weight, Grid3f vel, float dt)
		{
			buf.assign(src);

			AdvectBackwardColor(dst, src, vel, dt);

// 			AdvectBackwardColor(buf, src, vel, dt);
// 			AdvectBackwardColor(dst, buf, vel, -dt);
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(src.nx, src.ny, src.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 			K_UpdateBFECC << <gridDims, blockDims >> >(src, dst);
// 
// 			AdvectBackwardColor(dst, src, vel, dt);
// 
// 			K_Combine1stAnd2nd << <gridDims, blockDims >> >(dst, buf, 0.3f);

// 			buf.CopyFrom(dst);
// 			K_SmoothingBFECC << <gridDims, blockDims >> >(dst, buf, 0.95f);
		}

		__global__ void K_AdvectStaggeredBackward(Grid1f dst, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float dt, int axis)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			float vx, vy, vz;
			float fx, fy, fz;
			int ix, iy, iz;
			float w000, w100, w010, w001, w111, w011, w101, w110;

			if (axis == 1)
			{
				int i_minus = glm::clamp(i - 1, 0, nx - 2);
				int i_plus = glm::clamp(i, 0, nx - 2);

				vx = vel_u(i, j, k);
				vy = 0.25f*(vel_v(i_minus, j, k) + vel_v(i_minus, j + 1, k) + vel_v(i_plus, j, k) + vel_v(i_plus, j + 1, k));
				vz = 0.25f*(vel_w(i_minus, j, k) + vel_w(i_minus, j, k + 1) + vel_w(i_plus, j, k) + vel_w(i_plus, j, k + 1));
			}
			if (axis == 2)
			{
				int j_minus = glm::clamp(j - 1, 0, ny - 2);
				int j_plus = glm::clamp(j, 0, ny - 2);

				vx = 0.25f*(vel_u(i, j_minus, k) + vel_u(i + 1, j_minus, k) + vel_u(i, j_plus, k) + vel_u(i + 1, j_plus, k));
				vy = vel_v(i, j, k);
				vz = 0.25f*(vel_w(i, j_minus, k) + vel_w(i, j_minus, k + 1) + vel_w(i, j_plus, k) + vel_w(i, j_plus, k + 1));
			}
			if (axis == 3)
			{
				int k_minus = glm::clamp(k - 1, 0, nz - 2);
				int k_plus = glm::clamp(k, 0, nz - 2);

				vx = 0.25f*(vel_u(i, j, k_minus) + vel_u(i + 1, j, k_minus) + vel_u(i, j, k_plus) + vel_u(i + 1, j, k_plus));
				vy = 0.25f*(vel_v(i, j, k_minus) + vel_v(i, j + 1, k_minus) + vel_v(i, j, k_plus) + vel_v(i, j + 1, k_plus));
				vz = vel_w(i, j, k);
			}

			fx = i - vx*dt;
			fy = j - vy*dt;
			fz = k - vz*dt;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > nx - 1) fx = nx - 1.0f;
			if (fy < 0.0f) fy = 0.0f;
			if (fy > ny - 1) fy = ny - 1.0f;
			if (fz < 0.0f) fz = 0.0f;
			if (fz > nz - 1) fz = nz - 1.0f;

			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
			fx -= ix;			fy -= iy;			fz -= iz;

			if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
			if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
			if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx*(1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy*(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx*fy*fz;
			w011 = (1.0f - fx)*fy*fz;
			w101 = fx*(1.0f - fy)*fz;
			w110 = fx*fy*(1.0f - fz);

			int nxy = nx*ny;
			int k0 = ix + iy*nx + iz*nxy;

			dst(i, j, k) = w000*src[k0] + w100*src[k0 + 1] + w010*src[k0 + nx] + w001*src[k0 + nxy]
				+ w111*src[k0 + 1 + nx + nxy] + w011*src[k0 + nx + nxy] + w101*src[k0 + 1 + nxy] + w110*src[k0 + 1 + nx];
		}

		void PFKernel::AdvectStaggeredBackward(Grid1f dst, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float dt, int axis)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_AdvectStaggeredBackward << < gridDims, blockDims >> >(dst, src, vel_u, vel_v, vel_w, dt, axis);
		}

		__global__ void K_AdvectForward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			float fx, fy, fz;
			int ix, iy, iz;
			float w000, w100, w010, w001, w111, w011, w101, w110;
			
			auto vel_ijk = vel(i, j, k);

			fx = i + vel_ijk.x*dt;
			fy = j + vel_ijk.y*dt;
			fz = k + vel_ijk.z*dt;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > nx - 1) fx = nx - 1.0f;
			if (fy < 0.0f) fy = 0.0f;
			if (fy > ny - 1) fy = ny - 1.0f;
			if (fz < 0.0f) fz = 0.0f;
			if (fz > nz - 1) fz = nz - 1.0f;

			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
			fx -= ix;			fy -= iy;			fz -= iz;

			if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
			if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
			if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

			float val = src(i, j, k);
			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx*(1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy*(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx*fy*fz;
			w011 = (1.0f - fx)*fy*fz;
			w101 = fx*(1.0f - fy)*fz;
			w110 = fx*fy*(1.0f - fz);

			atomicAdd(&dst(ix, iy, iz), val*w000);
			atomicAdd(&dst(ix + 1, iy, iz), val*w100);
			atomicAdd(&dst(ix, iy + 1, iz), val*w010);
			atomicAdd(&dst(ix, iy, iz + 1), val*w001);
			atomicAdd(&dst(ix + 1, iy + 1, iz + 1), val*w111);
			atomicAdd(&dst(ix, iy + 1, iz + 1), val*w011);
			atomicAdd(&dst(ix + 1, iy, iz + 1), val*w101);
			atomicAdd(&dst(ix + 1, iy + 1, iz), val*w110);

// 			d(ix, iy, iz) += val*w000;
// 			d(ix + 1, iy, iz) += val*w100;
// 			d(ix, iy + 1, iz) += val*w010;
// 			d(ix, iy, iz + 1) += val*w001;
// 			d(ix + 1, iy + 1, iz + 1) += val*w111;
// 			d(ix, iy + 1, iz + 1) += val*w011;
// 			d(ix + 1, iy, iz + 1) += val*w101;
// 			d(ix + 1, iy + 1, iz) += val*w110;
		}

// 		__global__ void K_ClampVelocity(Grid3f vel)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			if (i >= vel.nx) return;
// 			if (j >= vel.ny) return;
// 			if (k >= vel.nz) return;
// 
// 			float mag_v = length(vel(i, j, k));
// 
// 			if (mag_v < 0.001f)
// 			{
// 				vel(i, j, k) = make_float3(0.0f);
// 			}
// 		}

		void PFKernel::AdvectForward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
		{
			dst.reset();

			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

//			LAUNCH_KERNEL(K_ClampVelocity, fDims, make_uint3(32, 8, 1), vel);

			K_AdvectForward << < gridDims, blockDims >> >(dst, src, vel, dt);
		}


		__global__ void K_AdvectForward(Grid4f dst, Grid4f src, Grid3f vel, Grid1f weight, float dt)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			float fx, fy, fz;
			int ix, iy, iz;
			float w000, w100, w010, w001, w111, w011, w101, w110;

			auto vel_ijk = vel(i, j, k);

			fx = i + vel_ijk.x*dt;
			fy = j + vel_ijk.y*dt;
			fz = k + vel_ijk.z*dt;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > nx - 1) fx = nx - 1.0f;
			if (fy < 0.0f) fy = 0.0f;
			if (fy > ny - 1) fy = ny - 1.0f;
			if (fz < 0.0f) fz = 0.0f;
			if (fz > nz - 1) fz = nz - 1.0f;

			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
			fx -= ix;			fy -= iy;			fz -= iz;

			if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
			if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
			if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

			auto val = src(i, j, k);
			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx*(1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy*(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx*fy*fz;
			w011 = (1.0f - fx)*fy*fz;
			w101 = fx*(1.0f - fy)*fz;
			w110 = fx*fy*(1.0f - fz);

			int nxy = nx*ny;
			int k0 = dst.index(ix, iy, iz);

			atomicAdd(&weight[k0], w000);
			atomicAdd(&weight[k0 + 1], w100);
			atomicAdd(&weight[k0 + nx], w010);
			atomicAdd(&weight[k0 + nxy], w001);
			atomicAdd(&weight[k0 + 1 + nx + nxy], w111);
			atomicAdd(&weight[k0 + nx + nxy], w011);
			atomicAdd(&weight[k0 + 1 + nxy], w101);
			atomicAdd(&weight[k0 + 1 + nx], w110);

			atomicAdd(&(dst[k0].x), w000*val.x);
			atomicAdd(&(dst[k0 + 1].x), w100*val.x);
			atomicAdd(&(dst[k0 + nx].x), w010*val.x);
			atomicAdd(&(dst[k0 + nxy].x), w001*val.x);
			atomicAdd(&(dst[k0 + 1 + nx + nxy].x), w111*val.x);
			atomicAdd(&(dst[k0 + nx + nxy].x), w011*val.x);
			atomicAdd(&(dst[k0 + 1 + nxy].x), w101*val.x);
			atomicAdd(&(dst[k0 + 1 + nx].x), w110*val.x);

			atomicAdd(&(dst[k0].y), w000*val.y);
			atomicAdd(&(dst[k0 + 1].y), w100*val.y);
			atomicAdd(&(dst[k0 + nx].y), w010*val.y);
			atomicAdd(&(dst[k0 + nxy].y), w001*val.y);
			atomicAdd(&(dst[k0 + 1 + nx + nxy].y), w111*val.y);
			atomicAdd(&(dst[k0 + nx + nxy].y), w011*val.y);
			atomicAdd(&(dst[k0 + 1 + nxy].y), w101*val.y);
			atomicAdd(&(dst[k0 + 1 + nx].y), w110*val.y);

			atomicAdd(&(dst[k0].z), w000*val.z);
			atomicAdd(&(dst[k0 + 1].z), w100*val.z);
			atomicAdd(&(dst[k0 + nx].z), w010*val.z);
			atomicAdd(&(dst[k0 + nxy].z), w001*val.z);
			atomicAdd(&(dst[k0 + 1 + nx + nxy].z), w111*val.z);
			atomicAdd(&(dst[k0 + nx + nxy].z), w011*val.z);
			atomicAdd(&(dst[k0 + 1 + nxy].z), w101*val.z);
			atomicAdd(&(dst[k0 + 1 + nx].z), w110*val.z);

			atomicAdd(&(dst[k0].w), w000*val.w);
			atomicAdd(&(dst[k0 + 1].w), w100*val.w);
			atomicAdd(&(dst[k0 + nx].w), w010*val.w);
			atomicAdd(&(dst[k0 + nxy].w), w001*val.w);
			atomicAdd(&(dst[k0 + 1 + nx + nxy].w), w111*val.w);
			atomicAdd(&(dst[k0 + nx + nxy].w), w011*val.w);
			atomicAdd(&(dst[k0 + 1 + nxy].w), w101*val.w);
			atomicAdd(&(dst[k0 + 1 + nx].w), w110*val.w);


			// 			d(ix, iy, iz) += val*w000;
			// 			d(ix + 1, iy, iz) += val*w100;
			// 			d(ix, iy + 1, iz) += val*w010;
			// 			d(ix, iy, iz + 1) += val*w001;
			// 			d(ix + 1, iy + 1, iz + 1) += val*w111;
			// 			d(ix, iy + 1, iz + 1) += val*w011;
			// 			d(ix + 1, iy, iz + 1) += val*w101;
			// 			d(ix + 1, iy + 1, iz) += val*w110;
		}

		__global__ void K_NormalizePigment(Grid4f dst, Grid1f weight)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int k0 = weight.index(i, j, k);
			float w = weight[k0];
			if (w > EPSILON)
			{
				dst[k0] /= w;
			}
			else
				dst[k0] = Vec4f(0.0f);
		}

		void PFKernel::AdvectForward(Grid4f dst, Grid4f src, Grid3f vel, Grid1f weight, float dt)
		{
			dst.reset();
			weight.reset();

			uint3 gDIms = make_uint3(vel.nx(), vel.ny(), vel.nz());
			uint3 bDims = make_uint3(32, 8, 1);

			LAUNCH_KERNEL(K_AdvectForward, gDIms, bDims, dst, src, vel, weight, dt);
			LAUNCH_KERNEL(K_NormalizePigment, gDIms, bDims, dst, weight);
		}

// 		__global__ void K_AdvectPigments(Grid3f pos, Grid1f mass, Grid3f vel, float dt)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx;
// 			int ny = mass.ny;
// 			int nz = mass.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int nxy = nx*ny;
// 
// 			int k0 = vel.Index(i, j, k);
// 
// 			int subId = pos.Index(i*MAX_PIGMENTS, j, k);
// 
// 			float w000, w100, w010, w001, w111, w011, w101, w110;
// 			int ix, iy, iz;
// 
// 			if (mass[k0] > MASS_TRESHOLD)
// 			{
// 				for (size_t t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					float3 p = pos[subId + t];
// 
// 					float fx = p.x - 0.5f;
// 					float fy = p.y - 0.5f;
// 					float fz = p.z - 0.5f;
// 
// 					if (fx < 0.0f) fx = 0.0f;
// 					if (fx > nx - 1) fx = nx - 1.0f;
// 					if (fy < 0.0f) fy = 0.0f;
// 					if (fy > ny - 1) fy = ny - 1.0f;
// 					if (fz < 0.0f) fz = 0.0f;
// 					if (fz > nz - 1) fz = nz - 1.0f;
// 
// 					ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
// 					fx -= ix;			fy -= iy;			fz -= iz;
// 
// 					if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
// 					if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
// 					if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }
// 
// 					w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
// 					w100 = fx*(1.0f - fy)*(1.0f - fz);
// 					w010 = (1.0f - fx)*fy*(1.0f - fz);
// 					w001 = (1.0f - fx)*(1.0f - fy)*fz;
// 					w111 = fx*fy*fz;
// 					w011 = (1.0f - fx)*fy*fz;
// 					w101 = fx*(1.0f - fy)*fz;
// 					w110 = fx*fy*(1.0f - fz);
// 
// 					float3 v = w000*vel[k0] + w100*vel[k0 + 1] + w010*vel[k0 + nx] + w001*vel[k0 + nxy]
// 						+ w111*vel[k0 + 1 + nx + nxy] + w011*vel[k0 + nx + nxy] + w101*vel[k0 + 1 + nxy] + w110*vel[k0 + 1 + nx];
// 
// 					pos[subId+t] += v*dt;
// 				}
// 			}
// 		}
// 
// 		void PFKernel::AdvectPigments(Grid3f pos, Grid1f mass, Grid3f vel, float dt)
// 		{
// 			LAUNCH_KERNEL(K_AdvectPigments, make_uint3(mass.nx, mass.ny, mass.nz), make_uint3(32, 8, 1), pos, mass, vel, dt);
// 		}


// 		__global__ void K_DepositPigments(Grid3f gPos, Grid3f prePos, Grid4f gColor, Grid4f preColor, Grid1i gNum, Grid1i preNum, Grid1f gAcc, Grid1f preAcc, Grid1u gMutex, Grid1f mass)
// 		{
// 			uint i = blockIdx.x * blockDim.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx();
// 			int ny = mass.ny();
// 			int nz = mass.nz();
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int ix, iy, iz;
// 			float fx, fy, fz;
// 			
// 			int id = prePos.Index(i*MAX_PIGMENTS, j, k);
// 			float m_ijk = mass(i, j, k);
// 
// 			if (m_ijk > MASS_TRESHOLD)
// 			{
// 				for (size_t t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					float3 pos = prePos[id + t];
// 					float4 color = preColor[id + t];
// 					float acc = preAcc[id + t];
// 
// // 					gPos[id + t] = pos;
// // 					gColor[id + t] = color;
// // 					gAcc[id + t] = acc;
// 
// 					ix = floor(pos.x);
// 					iy = floor(pos.y);
// 					iz = floor(pos.z);
// 
// 					if (ix < 0) { continue; }
// 					if (ix >= nx) { continue; }
// 					if (iy < 0) { continue; }
// 					if (iy >= ny) { continue; }
// 					if (iz < 0) { continue; }
// 					if (iz >= nz) { continue; }
// 
// 					fx = pos.x - ix;
// 					fy = pos.y - iy;
// 					fz = pos.z - iz;
// 
// 					int id_new = gPos.Index(ix*MAX_PIGMENTS, iy, iz);
// 
// 					int subId = GetPigmentIndex(make_float3(fx, fy, fz));
// 
// 					while (atomicCAS(&(gMutex[id_new + subId]), 0, 1) == 0) break;
// 					float acc_old = gAcc[id_new + subId];
// 					if (acc > acc_old)
// 					{
// 						gPos[id_new + subId] = pos;
// 						gColor[id_new + subId] = color;
// 						gAcc[id_new + subId] = acc;
// 						gNum[id_new + subId] = 1;
// 					}
// 					atomicExch(&(gMutex[id_new + subId]), 0);
// 				}
// 			}
// 			
// 		}

// 		__global__ void K_ClearCount(Grid3f gPos, Grid4f gColor, Grid1f gAcc, Grid1i gNum, Grid1u gMutex)
// 		{
// 			uint i = blockIdx.x * blockDim.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = gAcc.nx;
// 			int ny = gAcc.ny;
// 			int nz = gAcc.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int id = gAcc.Index(i, j, k);
// 
// 			gNum[id] = 0;
// 			gAcc[id] = -1.0f;
// 			gMutex[id] = 0;
// // 			gPos[id] = make_float3(0.0f);
// // 			gColor[id] = make_float4(0.0f);
// 		}

// 		void PFKernel::DepositParticles(Grid3f gPos, Grid3f prePos, Grid4f gColor, Grid4f preColor, Grid1i gNum, Grid1i preNum, Grid1f gAcc, Grid1f preAcc, Grid1u gMutex, Grid1f mass)
// 		{
// 			LAUNCH_KERNEL(K_ClearCount, make_uint3(gPos.nx, gPos.ny, gPos.nz), make_uint3(32, 8, 1), gPos, gColor, gAcc, gNum, gMutex);
// 
// // 			gPos.CopyFrom(prePos);
// // 			gColor.CopyFrom(preColor);
// // 			gNum.CopyFrom(preNum);
// // 			gAcc.CopyFrom(preAcc);
// 
// 			LAUNCH_KERNEL(K_DepositPigments, make_uint3(mass.nx, mass.ny, mass.nz), make_uint3(32, 8, 1), gPos, prePos, gColor, preColor, gNum, preNum, gAcc, preAcc, gMutex, mass);
// 		}

// 		__device__ void K_GetSubpixelColor(float4& pig, float& accuracy, Grid4f& color, Grid1f& acc, int i, int j, int k)
// 		{
// 			int pIdx = i / DIM_PIGMENT;
// 			int pIdy = j / DIM_PIGMENT;
// 			int pIdz = k / DIM_PIGMENT;
// 
// 			int sIdx = i % DIM_PIGMENT;
// 			int sIdy = j % DIM_PIGMENT;
// 			int sIdz = k % DIM_PIGMENT;
// 
// 			int id = color.Index(pIdx*MAX_PIGMENTS, pIdy, pIdz);
// 			int subId = GetPigmentIndex(sIdx, sIdy, sIdz);
// 
// 			pig = color[id+subId];
// 			accuracy = acc[id + subId];
// 		}
// 
// 		__global__ void K_SeedNewPigments(Grid3f gPos, Grid4f gColor, Grid4f preColor, Grid1i gNum, Grid1f gAcc, Grid1f preAcc, Grid1f mass, Grid3f vel, float dt)
// 		{
// 			uint i = blockIdx.x * blockDim.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx;
// 			int ny = mass.ny;
// 			int nz = mass.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int subNx = nx*DIM_PIGMENT;
// 			int subNy = ny*DIM_PIGMENT;
// 			int subNz = nz*DIM_PIGMENT;
// 
// 			float dist = 1.0f / DIM_PIGMENT;
// 
// 			int id = mass.Index(i, j, k);
// 			float3 vel_ijk = vel(i, j, k);
// 			float m_ijk = mass(i, j, k);
// 
// 			int subId = gPos.Index(i*MAX_PIGMENTS, j, k);
// 			//int num = counter[id];
// 
// 			//if (num < MAX_PIGMENTS)
// 
// 			if (m_ijk > MASS_TRESHOLD)
// 			{
// 				for (int t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					int num = gNum[subId+t];
// 					if (num == 0)
// 					{
// 						float3 fpos = make_float3(0.0f);
// 						SeedPigment(fpos, t, MAX_PIGMENTS);
// 
// 						fpos += make_float3(i, j, k);
// 
// 						int ix, iy, iz;
// 						float fx, fy, fz;
// 						float w000, w100, w010, w001, w111, w011, w101, w110;
// 
// 						fx = fpos.x - vel_ijk.x*dt;
// 						fy = fpos.y - vel_ijk.y*dt;
// 						fz = fpos.z - vel_ijk.z*dt;
// 
// 						fx = fx / dist - 0.5f;
// 						fy = fy / dist - 0.5f;
// 						fz = fz / dist - 0.5f;
// 
// 						ix = floor(fx);		iy = floor(fy);		iz = floor(fz);
// 						fx -= ix;			fy -= iy;			fz -= iz;
// 
// 						if (ix < 0) { ix = 0; fx = 0.0f; }
// 						if (ix >= subNx - 1) { ix = subNx - 2; fx = 1.0f; }
// 						if (iy < 0) { iy = 0; fy = 0.0f; }
// 						if (iy >= subNy - 1) { iy = subNy - 2; fy = 1.0f; }
// 						if (iz < 0) { iz = 0; fz = 0.0f; }
// 						if (iz >= subNz - 1) { iz = subNz - 2; fz = 1.0f; }
// 
// 						w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
// 						w100 = fx*(1.0f - fy)*(1.0f - fz);
// 						w010 = (1.0f - fx)*fy*(1.0f - fz);
// 						w001 = (1.0f - fx)*(1.0f - fy)*fz;
// 						w111 = fx*fy*fz;
// 						w011 = (1.0f - fx)*fy*fz;
// 						w101 = fx*(1.0f - fy)*fz;
// 						w110 = fx*fy*(1.0f - fz);
// 
// 
// 						float4 color = make_float4(0.0f);
// 						float acc = 0.0f;
// 						float alpha = 0.0f;
// 						float total_weight = 0.0f;
// 
// 						//						color += K_GetSubpixelColor(pre_pig, ix, iy, iz);
// 						float4 t_pig;
// 						float t_acc;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix, iy, iz);
// 						alpha = w000*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix + 1, iy, iz);
// 						alpha = w100*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix, iy + 1, iz);
// 						alpha = w010*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix, iy, iz + 1);
// 						alpha = w001*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix + 1, iy + 1, iz + 1);
// 						alpha = w111*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix, iy + 1, iz + 1);
// 						alpha = w011*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix + 1, iy, iz + 1);
// 						alpha = w101*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						K_GetSubpixelColor(t_pig, t_acc, preColor, preAcc, ix + 1, iy + 1, iz);
// 						alpha = w110*t_pig.w;		total_weight += alpha;
// 						color += t_pig*alpha;	acc += t_acc*alpha;
// 
// 						if (total_weight > EPSILON)
// 						{
// 							color /= total_weight;
// 							acc /= total_weight;
// 						}
// 						else
// 						{
// 							color = make_float4(0.0f);
// 							acc = 0.0f;
// 						}
// 
// 						gPos[subId + t] = fpos;
// 						gColor[subId+t] = color;
// 						gAcc[subId+t] = acc*max(fx, 1.0f - fx)*max(fy, 1.0f - fy)*max(fz, 1.0f - fz);
// 
// 						// 						if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
// 						// 						if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
// 						// 						if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }
// 
// 						// 						w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
// 						// 						w100 = fx*(1.0f - fy)*(1.0f - fz);
// 						// 						w010 = (1.0f - fx)*fy*(1.0f - fz);
// 						// 						w001 = (1.0f - fx)*(1.0f - fy)*fz;
// 						// 						w111 = fx*fy*fz;
// 						// 						w011 = (1.0f - fx)*fy*fz;
// 						// 						w101 = fx*(1.0f - fy)*fz;
// 						// 						w110 = fx*fy*(1.0f - fz);
// 
// 						// 						int nxy = nx*ny;
// 						// 						int k0 = ix + iy*nx + iz*nxy;
// 
// 						// 						float4 color = w000*pigOnCenter[k0] + w100*pigOnCenter[k0 + 1] + w010*pigOnCenter[k0 + nx] + w001*pigOnCenter[k0 + nxy]
// 						// 							+ w111*pigOnCenter[k0 + 1 + nx + nxy] + w011*pigOnCenter[k0 + nx + nxy] + w101*pigOnCenter[k0 + 1 + nxy] + w110*pigOnCenter[k0 + 1 + nx];
// 					}
// 
// 					//					pigOnCenter(i, j, k) = color;
// 				}
// 
// 
// 				/*				for (int t = 0; t < num; t++)
// 				{
// 				float3 pos = pigments[id].pos[t];
// 				if (pos.x < i || pos.x > i + 1 || pos.y < j || pos.y > j + 1 || pos.z < k || pos.z > k + 1)
// 				printf("##################");
// 				}*/
// 			}
// 			else
// 			{
// 				for (int t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					gNum[subId+t] = 0;
// 					gAcc[subId+t] = -1.0f;
// //					gColor[subId+t] = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
// 				}
// 			}
// 		}
// 
// 		void PFKernel::SeedNewParticles(Grid3f gPos, Grid4f gColor, Grid4f preColor, Grid1i gNum, Grid1f gAcc, Grid1f preAcc, Grid1f mass, Grid3f vel, float dt)
// 		{
// 			LAUNCH_KERNEL(K_SeedNewPigments, make_uint3(mass.nx, mass.ny, mass.nz), make_uint3(32, 8, 1), gPos, gColor, preColor, gNum, gAcc, preAcc, mass, vel, dt);
// 		}

		/*		__global__ void K_RasterizePigments2Node(Grid4f nodes, GridPig cvPigments, Grid1i counter, uint3 winOrigin)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = nodes.nx;
			int ny = nodes.ny;
			int nz = nodes.nz;

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int cv_nx = cvPigments.nx;
			int cv_ny = cvPigments.ny;
			int cv_nz = cvPigments.nz;

			winOrigin = make_uint3(0, 0, 0);

			int cv_i = winOrigin.x + i;
			int cv_j = winOrigin.y + j;
			int cv_k = winOrigin.z + k;

			int ix, iy, iz;
			int cvId;
			int num;

			float4 total_color = make_float4(0.0f);
			float total_weight = 0.0f;

			ix = winOrigin.x + i;		iy = winOrigin.y + j;		iz = winOrigin.z + k;
			if (ix < cv_nx && iy < cv_ny && iz < cv_nz)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);
				
				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i - 1;	iy = winOrigin.y + j;		iz = winOrigin.z + k;
			if (ix >= 0 && iy < cv_ny && iz < cv_nz)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i;		iy = winOrigin.y + j - 1;	iz = winOrigin.z + k;
			if (ix < cv_nx && iy >= 0 && iz < cv_nz)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i;		iy = winOrigin.y + j;		iz = winOrigin.z + k - 1;
			if (ix < cv_nx && iy < cv_ny && iz >= 0)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i - 1;		iy = winOrigin.y + j - 1;	iz = winOrigin.z + k;
			if (ix >= 0 && iy >= 0 && iz < cv_nz)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i;		iy = winOrigin.y + j - 1;	iz = winOrigin.z + k - 1;
			if (ix < cv_nx && iy >= 0 && iz >= 0)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i - 1;	iy = winOrigin.y + j;		iz = winOrigin.z + k - 1;
			if (ix >= 0 && iy < cv_ny && iz >= 0)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			ix = winOrigin.x + i - 1;	iy = winOrigin.y + j - 1;	iz = winOrigin.z + k - 1;
			if (ix >= 0 && iy >= 0 && iz >= 0)
			{
				cvId = cvPigments.Index(ix, iy, iz);
				num = counter(ix, iy, iz);
				num = min(num, MAX_PIGMENTS);

				for (int t = 0; t < num; t++)
				{
					float3 pgPos = cvPigments[cvId].pos[t];
					float4 pgColor = cvPigments[cvId].color[t];
					float weight = 1.0f;
					float fx = 1.0f - abs(pgPos.x - cv_i);
					float fy = 1.0f - abs(pgPos.y - cv_j);
					float fz = 1.0f - abs(pgPos.z - cv_k);
//					if (fx*fy*fz < 0)
//						printf("%f \n", fx*fy*fz);
					weight *= fx*fy*fz;
					total_weight += weight;
					total_color += weight*pgColor;
				}
			}

			if (total_weight > EPSILON)
			{
				nodes(i, j, k) = total_color / total_weight;
			}
			else
			{
				nodes(i, j, k) = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
			}
		}

		__global__ void K_InterpolatePigments2Center(Grid4f centers, Grid4f nodes)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = centers.nx;
			int ny = centers.ny;
			int nz = centers.nz;

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			float4 pig = make_float4(0.0f);

			pig += nodes(i, j, k);
			pig += nodes(i+1, j, k);
			pig += nodes(i, j+1, k);
			pig += nodes(i, j, k+1);
			pig += nodes(i+1, j+1, k);
			pig += nodes(i, j+1, k+1);
			pig += nodes(i+1, j, k+1);
			pig += nodes(i+1, j+1, k+1);

			centers(i, j, k) = pig / 8.0;
		}*/

// 		__global__ void K_RasterizePigments2Center(Grid4f pgOnCenter, Grid3f gPos, Grid4f gColor)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = pgOnCenter.nx;
// 			int ny = pgOnCenter.ny;
// 			int nz = pgOnCenter.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int id = pgOnCenter.Index(i, j, k);
// 			int subId = gColor.Index(i*MAX_PIGMENTS, j, k);
// 
// 			float4 color = make_float4(0.0f);
// 			float total_weight = 0.0f;
// 			for (int t = 0; t < MAX_PIGMENTS; t++)
// 			{
// 				float3 pos = gPos[subId + t];
// 				float4 subColor = gColor[subId + t];
// 				float weight = subColor.w;
// 
// 				total_weight += weight;
// 				color += weight*subColor;
// 			}
// 
// 			if (total_weight > EPSILON)
// 			{
// 				pgOnCenter[id] = color / total_weight;
// // 				pgOnCenter[id] /= pgOnCenter[id].w;
// // 				pgOnCenter[id].w = 1.0f;
// 			}
// 			else
// 			{
// 				pgOnCenter[id] = make_float4(0.0f);
// 			}
// 		}
// 
// 
// 		void PFKernel::RasterizePigments2Grid(Grid4f pgOnCenter, Grid3f pos, Grid4f color, Grid1i counter, uint3 winOrigin)
// 		{
// //			LAUNCH_KERNEL(K_RasterizePigments2Node, make_uint3(pgOnNode.nx, pgOnNode.ny, pgOnNode.nz), make_uint3(32, 8, 1), pgOnNode, pgOnParticle, counter, winOrigin);
// 
// //			LAUNCH_KERNEL(K_InterpolatePigments2Center, make_uint3(pgOnCenter.nx, pgOnCenter.ny, pgOnCenter.nz), make_uint3(32, 8, 1), pgOnCenter, pgOnNode);
// 
// 			LAUNCH_KERNEL(K_RasterizePigments2Center, make_uint3(pgOnCenter.nx, pgOnCenter.ny, pgOnCenter.nz), make_uint3(32, 8, 1), pgOnCenter, pos, color);
// 		}

// 		__global__ void K_ExtraploateColor(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = dst.nx;
// 			int ny = dst.ny;
// 			int nz = dst.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int id0 = mass.Index(i, j, k);
// 
// 			float m0 = pre_mass[id0];
// 
// 			float rho_threshold = 0.05f;
// 
// 			if (m0 > MASS_THESHOLD2)
// 			{
// 				dst[id0] = src[id0];
// 				mass[id0] = pre_mass[id0];
// 			}
// 			else
// 			{
// 				for (int di = -1; di <= 1; di++){
// 					for (int dj = -1; dj <= 1; dj++){
// 						for (int dk = -1; dk <= 1; dk++)
// 						{
// 							int ix = i + di;
// 							int iy = j + dj;
// 							int iz = k + dk;
// 
// 							ix = clamp(ix, 0, nx - 1);
// 							iy = clamp(iy, 0, ny - 1);
// 							iz = clamp(iz, 0, nz - 1);
// 
// 							int id1 = pre_mass.Index(ix, iy, iz);
// 
// 							float m1 = pre_mass[id1];
// 
// 							if (m1 > MASS_THESHOLD2)
// 							{
// 								dst[id0] = src[id1];
// 								mass[id0] = pre_mass[id1];
// 								break;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		__global__ void InitalBottomLayer(Grid1f mass)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx();
// 			int ny = mass.ny();
// 			int nz = mass.nz();
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			if (k == 0)
// 			{
// 				mass(i, j, k) = MASS_THESHOLD2 + 0.1f;
// 			}
// 		}


// 		void PFKernel::ExtraploateColor(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, int iteration)
// 		{
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(src.nx, src.ny, src.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			for (size_t i = 0; i < iteration; i++)
// 			{
// 				K_ExtraploateColor << < gridDims, blockDims >> >(dst, src, mass, pre_mass);
// 				mass.Swap(pre_mass);
// 				dst.Swap(src);
// 			}
// 
// 			dst.Swap(src);
// 		}

// 		__global__ void K_ExtraploateMass(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, float* density_brush, int3 simOrigin, int3 brushOrigin)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = dst.nx;
// 			int ny = dst.ny;
// 			int nz = dst.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int ix = i + simOrigin.x - brushOrigin.x;
// 			int iy = j + simOrigin.y - brushOrigin.y;
// 			int iz = k + simOrigin.z - brushOrigin.z;
// 
// 			if (ix < 0 || ix >= nx)	return;
// 			if (iy < 0 || iy >= nx)	return;
// 			if (iz < 0 || iz >= nx)	return;
// 
// 			int bId = mass.Index(ix, iy, iz);
// 			float rho = density_brush[bId];
// 
// 			int id0 = mass.Index(i, j, k);
// 
// 			float m0 = pre_mass[id0];
// 
// 			float rho_threshold = 0.05f;
// 
// 			if (m0 > MASS_THESHOLD2)
// 			{
// 				dst[id0] = src[id0];
// 				mass[id0] = pre_mass[id0];
// 			}
// 			else
// 			{
// 				for (int di = -1; di <= 1; di++){
// 					for (int dj = -1; dj <= 1; dj++){
// 						for (int dk = -1; dk <= 1; dk++)
// 						{
// 							int ix = i + di;
// 							int iy = j + dj;
// 							int iz = k + dk;
// 
// 							ix = clamp(ix, 0, nx - 1);
// 							iy = clamp(iy, 0, ny - 1);
// 							iz = clamp(iz, 0, nz - 1);
// 
// 							int id1 = pre_mass.Index(ix, iy, iz);
// 
// 							float m1 = pre_mass[id1];
// 
// 							if (m1 > MASS_THESHOLD2 && rho > 0.01f)
// 							{
// 								dst[id0] = src[id1];
// 								mass[id0] = pre_mass[id1];
// 								break;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 
// 		void PFKernel::ExtraploateMass(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, float* density_brush, int3 simOrigin, int3 brushOrigin, int iteration)
// 		{
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(src.nx, src.ny, src.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			InitalBottomLayer << < gridDims, blockDims >> >(pre_mass);
// 
// 			for (size_t i = 0; i < iteration; i++)
// 			{
// 				K_ExtraploateMass << < gridDims, blockDims >> >(dst, src, mass, pre_mass, density_brush, simOrigin, brushOrigin);
// 				mass.Swap(pre_mass);
// 				dst.Swap(src);
// 			}
// 
// 			mass.Swap(pre_mass);
// 			dst.Swap(src);
// 		}

// 		__global__ void K_ExtraploateColor(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, Grid3f pos, Grid4f color, Grid1i num, Grid1f accuracy)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = dst.nx;
// 			int ny = dst.ny;
// 			int nz = dst.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int id0 = mass.Index(i, j, k);
// 			int subId = pos.Index(i*MAX_PIGMENTS, j, k);
// 
// 			float m0 = pre_mass[id0];
// 
// 			if (m0 > MASS_THESHOLD2)
// 			{
// 				dst[id0] = src[id0];
// 				mass[id0] = pre_mass[id0];
// 			}
// 			else
// 			{
// 				float maxM = -100.0f;
// 				float minD = 100.0f;
// 				for (int di = -1; di <= 1; di++){
// 					for (int dj = -1; dj <= 1; dj++){
// 						for (int dk = -1; dk <= 1; dk++)
// 						{
// 							int ix = i + di;
// 							int iy = j + dj;
// 							int iz = k + dk;
// 
// 							ix = clamp(ix, 0, nx - 1);
// 							iy = clamp(iy, 0, ny - 1);
// 							iz = clamp(iz, 0, nz - 1);
// 
// 							float3 fd = make_float3(ix - i, iy - j, iz - k);
// 							float len = length(fd);
// 
// 							int id1 = pre_mass.Index(ix, iy, iz);
// 
// 							float m1 = pre_mass[id1];
// 
// 							bool replace = false;
// 
// 							if (m1 > maxM)
// 							{
// 								replace = true;
// 							}
// 							else if (len < minD && m1 >(maxM - 0.01f))
// 							{
// 								replace = true;
// 							}
// 
// 							if (m1 > maxM)
// 							{
// 								maxM = m1;
// 							}
// 
// 							if (len < minD)
// 							{
// 								minD = len;
// 							}
// 
// 							if (m1 > MASS_THESHOLD2 && replace && len > 0.1f)
// 							{
// 								float4 clr = src[id1];
// 								dst[id0] = clr;
// 
// 								for (int t = 0; t < MAX_PIGMENTS; t++)
// 								{
// 									color[subId + t] = clr;
// 									num[subId+t] = 1;
// 									accuracy[subId+t] = 0.5f;
// 									float3 fpos;
// 									SeedPigment(fpos, t, MAX_PIGMENTS);
// 
// 									pos[subId+t] = fpos + make_float3(i, j, k);
// 								}
// 
// 								mass[id0] = pre_mass[id1];
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		void PFKernel::ExtraploateColor(Grid4f dst, Grid4f src, Grid1f mass, Grid1f pre_mass, Grid3f pos, Grid4f color, Grid1i num, Grid1f accuracy, int iteration)
// 		{
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(src.nx, src.ny, src.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			for (size_t i = 0; i < iteration; i++)
// 			{
// 				K_ExtraploateColor << < gridDims, blockDims >> >(dst, src, mass, pre_mass, pos, color, num, accuracy);
// 				mass.Swap(pre_mass);
// 				dst.Swap(src);
// 			}
// 
// 			mass.Swap(pre_mass);
// 			dst.Swap(src);
// 		}

		__global__ void K_AdvectBackward(Grid3f dst, Grid3f src, Grid3f vel, float dt)
		{
			uint i = blockDim.x * blockIdx.x + threadIdx.x;
			uint j = blockIdx.y * blockDim.y + threadIdx.y;
			uint k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			auto vel_ijk = vel(i, j, k);

			int ix, iy, iz;
			float fx, fy, fz;
			float w000, w100, w010, w001, w111, w011, w101, w110;

			fx = i - vel_ijk.x*dt;
			fy = j - vel_ijk.y*dt;
			fz = k - vel_ijk.z*dt;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > nx - 1) fx = nx - 1.0f;
			if (fy < 0.0f) fy = 0.0f;
			if (fy > ny - 1) fy = ny - 1.0f;
			if (fz < 0.0f) fz = 0.0f;
			if (fz > nz - 1) fz = nz - 1.0f;

			ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
			fx -= ix;			fy -= iy;			fz -= iz;

			if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
			if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
			if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx*(1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy*(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx*fy*fz;
			w011 = (1.0f - fx)*fy*fz;
			w101 = fx*(1.0f - fy)*fz;
			w110 = fx*fy*(1.0f - fz);

			int nxy = nx*ny;
			int k0 = ix + iy*nx + iz*nxy;

			dst(i, j, k) = w000*src[k0] + w100*src[k0 + 1] + w010*src[k0 + nx] + w001*src[k0 + nxy]
				+ w111*src[k0 + 1 + nx + nxy] + w011*src[k0 + nx + nxy] + w101*src[k0 + 1 + nxy] + w110*src[k0 + 1 + nx];
		}

		void PFKernel::AdvectBackward(Grid3f dst, Grid3f src, Grid3f vel, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_AdvectBackward << < gridDims, blockDims >> >(dst, src, vel, dt);
		}

		__device__ float K_SharpeningWeight(float dist)
		{
			float fx = dist - floor(dist);
			fx = 1.0f - 2.0f*abs(fx - 0.5f);

			if (fx < 0.01f)
			{
				fx = 0.0f;
			}

			return fx;
		}

		__global__ void K_Sharpening(
			Grid1f dst, 
			Grid3f dir, 
			Grid1f src, 
			Grid1f vel_u, 
			Grid1f vel_v, 
			Grid1f vel_w,
			Grid1f omega,
			float gamma,
			float h,
			float dt)
		{
			const int i = blockDim.x * blockIdx.x + threadIdx.x;
			const int j = blockIdx.y * blockDim.y + threadIdx.y;
			const int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int i_minus = glm::clamp(i - 1, 0, nx - 1);
			int i_plus = glm::clamp(i + 1, 0, nx - 1);

			int j_minus = glm::clamp(j - 1, 0, ny - 1);
			int j_plus = glm::clamp(j + 1, 0, ny - 1);

			int k_minus = glm::clamp(k - 1, 0, nz - 1);
			int k_plus = glm::clamp(k + 1, 0, nz - 1);

// 			float norm_x, norm_y, norm_z;
// 
// 			norm_x = src(i_plus, j, k) - src(i_minus, j, k);
// 			norm_y = src(i, j_plus, k) - src(i, j_minus, k);
// 			norm_z = src(i, j, k_plus) - src(i, j, k_minus);
// 
// 			float l = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z);
// 			if (l < EPSILON)
// 			{
// 				norm_x = 0.0f;
// 				norm_y = 0.0f;
// 				norm_z = 0.0f;
// 			}
// 			else
// 			{
// 				norm_x /= l;
// 				norm_y /= l;
// 				norm_z /= l;
// 			}
// 			
// 
// 			dir(i, j, k) = make_float3(norm_x, norm_y, norm_z);
// 			dst(i, j, k) = src(i, j, k);

//			__syncthreads();

			int k0, k1;
			Vec3f n0, n1;
			float c1, c0, dc;

			float ceo = 24.0f * gamma / h;
			k0 = dir.index(i, j, k);

			float weight = 0.0f;


/*			if (i < nx - 1)
			{
				k1 = dir.Index(i_plus, j, k);
				n0 = dir[k0];
				n1 = dir[k1];

				c1 = src[k1] * (1.0f - src[k1])*n1.x / omega[k1];
				c0 = src[k0] * (1.0f - src[k0])*n0.x / omega[k0];

				dc = 0.5f*(c1 + c0)*ceo*dt;

				atomicAdd(&(dst[k0]), -dc);
				atomicAdd(&(dst[k1]), dc);
				//dst[k0] -= dc;
				//dst[k1] += dc;
			}


			//j and j+1
			if (j < ny - 1)
			{
				k1 = dir.Index(i, j_plus, k);
				n0 = dir[k0];
				n1 = dir[k1];

				c1 = src[k1] * (1.0f - src[k1])*n1.y / omega[k1];
				c0 = src[k0] * (1.0f - src[k0])*n0.y / omega[k0];

				dc = 0.5f*(c1 + c0)*ceo*dt;

				atomicAdd(&(dst[k0]), -dc);
				atomicAdd(&(dst[k1]), dc);
				//dst[k0] -= dc;
				//dst[k1] += dc;
			}

			//k and k+1
			if (k < nz - 1)
			{
				k1 = dir.Index(i, j, k_plus);
				n0 = dir[k0];
				n1 = dir[k1];

				c1 = src[k1] * (1.0f - src[k1])*n1.z / omega[k1];
				c0 = src[k0] * (1.0f - src[k0])*n0.z / omega[k0];

				dc = 0.5f*(c1 + c0)*ceo*dt;

				atomicAdd(&(dst[k0]), -dc);
				atomicAdd(&(dst[k1]), dc);
				//dst[k0] -= dc;
				//dst[k1] += dc;
			}
*/

			//-----------------------------------------------------------------------
			//i and i+1
			
			if (i < nx - 1)
			{
				k1 = dir.index(i_plus, j, k);
				//if (src[k0] < 1.0f && src[k1] < 1.0f)
				{
					n0 = dir[k0];
					n1 = dir[k1];

					c1 = src[k1] * (1.0f - src[k1])*n1.x / omega[k1];
					c0 = src[k0] * (1.0f - src[k0])*n0.x / omega[k0];

					dc = 0.5f*(c1 + c0)*ceo*dt;

					weight = K_SharpeningWeight(vel_u(i + 1, j, k)*dt / h);
					atomicAdd(&(dst[k0]), -weight*dc);
					atomicAdd(&(dst[k1]), weight*dc);

// 					weight = K_SharpeningWeight(vel_u(i+1, j, k));
// 
// 					dst[k0] -= weight*dc;
				}
				
			}


			//j and j+1
			if (j < ny - 1)
			{
				k1 = dir.index(i, j_plus, k);
				//if (src[k0] < 1.0f && src[k1] < 1.0f)
				{
					n0 = dir[k0];
					n1 = dir[k1];

					c1 = src[k1] * (1.0f - src[k1])*n1.y / omega[k1];
					c0 = src[k0] * (1.0f - src[k0])*n0.y / omega[k0];

					dc = 0.5f*(c1 + c0)*ceo*dt;

					weight = K_SharpeningWeight(vel_v(i, j + 1, k)*dt / h);
	 				atomicAdd(&(dst[k0]), -weight*dc);
					atomicAdd(&(dst[k1]), weight*dc);
// 					weight = K_SharpeningWeight(vel_v(i, j + 1, k));
// 
// 					dst[k0] -= weight*dc;
					//dst[k1] += dc;
				}
			}

			//k and k+1
			if (k < nz - 1)
			{
				k1 = dir.index(i, j, k_plus);
				//if (src[k0] < 1.0f && src[k1] < 1.0f)
				{
					n0 = dir[k0];
					n1 = dir[k1];

					c1 = src[k1] * (1.0f - src[k1])*n1.z / omega[k1];
					c0 = src[k0] * (1.0f - src[k0])*n0.z / omega[k0];

					dc = 0.5f*(c1 + c0)*ceo*dt;

					weight = K_SharpeningWeight(vel_w(i, j, k + 1)*dt / h);
					atomicAdd(&(dst[k0]), -weight*dc);
					atomicAdd(&(dst[k1]), weight*dc);
// 					weight = K_SharpeningWeight(vel_w(i, j, k + 1));
// 
// 					dst[k0] -= weight*dc;
					//dst[k1] += dc;
				}
 			}

			/////////

/*			if (i > 0)
			{
				k1 = dir.Index(i_minus, j, k);
				if (src[k0] < 1.0f && src[k1] < 1.0f)
				{
					n0 = dir[k0];
					n1 = dir[k1];

					c1 = src[k1] * (1.0f - src[k1])*n1.x / omega[k1];
					c0 = src[k0] * (1.0f - src[k0])*n0.x / omega[k0];

					dc = 0.5f*(c1 + c0)*ceo*dt;

					// 				atomicAdd(&(dst[k0]), -dc);
					// 				atomicAdd(&(dst[k1]), dc);
					weight = K_SharpeningWeight(vel_u(i, j, k));

					dst[k0] += weight*dc;
					//dst[k1] += dc;
				}
			}


			//j and j+1
			if (j > 0)
			{
				k1 = dir.Index(i, j_minus, k);
				if (src[k0] < 1.0f && src[k1] < 1.0f)
				{
					n0 = dir[k0];
					n1 = dir[k1];

					c1 = src[k1] * (1.0f - src[k1])*n1.y / omega[k1];
					c0 = src[k0] * (1.0f - src[k0])*n0.y / omega[k0];

					dc = 0.5f*(c1 + c0)*ceo*dt;

					// 				atomicAdd(&(dst[k0]), -dc);
					// 				atomicAdd(&(dst[k1]), dc);
					weight = K_SharpeningWeight(vel_v(i, j, k));
					dst[k0] += weight*dc;
					//dst[k1] += dc;
				}
			}

			//k and k+1
			if (k > 0)
			{
				k1 = dir.Index(i, j, k_minus);
				if (src[k0] < 1.0f && src[k1] < 1.0f)
				{
					n0 = dir[k0];
					n1 = dir[k1];

					c1 = src[k1] * (1.0f - src[k1])*n1.z / omega[k1];
					c0 = src[k0] * (1.0f - src[k0])*n0.z / omega[k0];

					dc = 0.5f*(c1 + c0)*ceo*dt;

					// 				atomicAdd(&(dst[k0]), -dc);
					// 				atomicAdd(&(dst[k1]), dc);
					weight = K_SharpeningWeight(vel_w(i, j, k));
					dst[k0] += weight*dc;
					//dst[k1] += dc;
				}
			}*/

		}

		__global__ void K_ComputeNormals(Grid1f dst, Grid3f dir, Grid1f src)
		{
			const int i = blockDim.x * blockIdx.x + threadIdx.x;
			const int j = blockIdx.y * blockDim.y + threadIdx.y;
			const int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = src.nx();
			int ny = src.ny();
			int nz = src.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int i_minus = glm::clamp(i - 1, 0, nx - 1);
			int i_plus = glm::clamp(i + 1, 0, nx - 1);

			int j_minus = glm::clamp(j - 1, 0, ny - 1);
			int j_plus = glm::clamp(j + 1, 0, ny - 1);

			int k_minus = glm::clamp(k - 1, 0, nz - 1);
			int k_plus = glm::clamp(k + 1, 0, nz - 1);

			float norm_x, norm_y, norm_z;

			norm_x = src(i_plus, j, k) - src(i_minus, j, k);
			norm_y = src(i, j_plus, k) - src(i, j_minus, k);
			norm_z = src(i, j, k_plus) - src(i, j, k_minus);

			float l = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z);
			if (l < EPSILON)
			{
				norm_x = 0.0f;
				norm_y = 0.0f;
				norm_z = 0.0f;
			}
			else
			{
				norm_x /= l;
				norm_y /= l;
				norm_z /= l;
			}

			dir(i, j, k) = Vec3f(norm_x, norm_y, norm_z);
			dst(i, j, k) = src(i, j, k);
		}

		void PFKernel::Sharpening(Grid1f dst, Grid3f dir, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f omega, float gamma,
			float h, float dt)
		{
			dst.reset();

			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
			K_ComputeNormals << < gridDims, blockDims >> >(dst, dir, src);
			K_Sharpening << < gridDims, blockDims >> >(dst, dir, src, vel_u, vel_v, vel_w, omega, gamma, h, dt);
		}

		__device__ float K_MapDiffusion(float m, float mag_v)
		{
			float weight = 1.0f;
//			if (mag_v < 0.001f && m < 0.1f)
// 			{
// 				weight = 0.0f;
// 			}
			if (m > 1.0f || m < 0.0f)	return 100.0f*weight;
			else return 0.0f;
		}

		__global__ void K_JacobiStep(Grid1f dst, Grid1f src, Grid1f buf, Grid3f vel, float a, float c)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int k0 = src.index(i, j, k);
			int nxy = nx*ny;

// 			float c1 = 1.0f / c;
// 			float c2 = (1.0f - c1) / 6.0f;

			float mag_v = vel(i, j, k).norm();

			int i_minus = glm::clamp(i - 1, 0, nx - 1);
			int i_plus = glm::clamp(i + 1, 0, nx - 1);

			int j_minus = glm::clamp(j - 1, 0, ny - 1);
			int j_plus = glm::clamp(j + 1, 0, ny - 1);

			int k_minus = glm::clamp(k - 1, 0, nz - 1);
			int k_plus = glm::clamp(k + 1, 0, nz - 1);

			float m_ijk = src[k0];

			float ax0 = a*K_MapDiffusion(0.5f*(m_ijk + src(i_minus, j, k)), mag_v);
			float ax1 = a*K_MapDiffusion(0.5f*(m_ijk + src(i_plus, j, k)), mag_v);

			float ay0 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j_minus, k)), mag_v);
			float ay1 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j_plus, k)), mag_v);

			float az0 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j, k_minus)), mag_v);
			float az1 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j, k_plus)), mag_v);

			float c1 = 1.0f / (1.0f + ax0 + ax1 + ay0 + ay1 + az0 + az1);

			//dst[k0] = (c1*src[k0] + c2*(buf(i_plus, j, k) + buf(i_minus, j, k) + buf(i, j_plus, k) + buf(i, j_minus, k) + buf(i, j, k_plus) + buf(i, j, k_minus)));
			dst[k0] = (c1*src[k0] + c1*ax1*buf(i_plus, j, k) + c1*ax0*buf(i_minus, j, k) + c1*ay1*buf(i, j_plus, k) + c1*ay0*buf(i, j_minus, k) + c1*az1*buf(i, j, k_plus) + c1*az0*buf(i, j, k_minus));
		}

		void PFKernel::Jacobi(Grid1f dst, Grid1f src, Grid1f buf, Grid3f vel, float a, float c, int iteration)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_CopyData << < gridDims, blockDims >> >(dst, src);
			for (int i = 0; i < iteration; i++)
			{
				K_CopyData << < gridDims, blockDims >> >(buf, dst);
				K_JacobiStep << < gridDims, blockDims >> >(dst, src, buf, vel, a, c);
			}
		}

		__device__ float D_VisocityFromShearingRate(float rate)
		{
			if (rate > 0.2f)
			{
				return MU_0;
			}
			else if (rate < 0.01f)
			{
				return MU_INF;
			}
			else
			{
				return MU_INF + (MU_0 - MU_INF)*(rate - 0.01f) / 0.19f;
			}
			//return MU_0 + (MU_INF - MU_0)*pow(1.0f + pow(SCALING*rate, ALPHA), (EXP_N - 1.0f) / 0.1f);
		}

		__global__ void K_SetVelocityBoundary(Grid3f vel)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = vel.nx();
			int ny = vel.ny();
			int nz = vel.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			if (i == 0) { vel(i, j, k) = Vec3f(0.0f); return; }
			if (i == nx - 1) { vel(i, j, k) = Vec3f(0.0f); return; }

			if (j == 0) { vel(i, j, k) = Vec3f(0.0f); return; }
			if (j == ny - 1) { vel(i, j, k) = Vec3f(0.0f); return; }

			if (k == 0) { vel(i, j, k) = Vec3f(0.0f); return; }
			if (k == nz - 1) { vel(i, j, k) = vel(i, j, k-1); return; }
		}

		__global__ void K_Laplacian(Grid3f dst, Grid3f src, Grid3f buf, Grid1f mass, float a)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = dst.nx();
			int ny = dst.ny();
			int nz = dst.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			int k0 = src.index(i, j, k);
			int nxy = nx*ny;

			float m = mass[k0];

			m = m > 1.0f ? 1.0f : m;
			m = m < 0.0f ? 0.0f : m;
			float vis = (VIS1*m + VIS2*(1.0f - m));

//			if (k > nz*0.75f)
//			{
//				vis = 0.0f;
//			}

			float c0 = 1.0f + 6.0f*a*vis;

			float c1 = 1.0f / c0;
			float c2 = (1.0f - c1) / 6.0f;

			int i_minus = glm::clamp(i - 1, 0, nx - 1);
			int i_plus = glm::clamp(i + 1, 0, nx - 1);

			int j_minus = glm::clamp(j - 1, 0, ny - 1);
			int j_plus = glm::clamp(j + 1, 0, ny - 1);

			int k_minus = glm::clamp(k - 1, 0, nz - 1);
			int k_plus = glm::clamp(k + 1, 0, nz - 1);


			dst[k0] = (c1*src[k0] + c2*(buf(i_plus, j, k) + buf(i_minus, j, k) + buf(i, j_plus, k) + buf(i, j_minus, k) + buf(i, j, k_plus) + buf(i, j, k_minus)));
		}

		void PFKernel::ApplyViscosity(Grid3f dst, Grid3f src, Grid3f buf, Grid1f mass, float a, int iteration)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_CopyData << < gridDims, blockDims >> >(dst, src);
			K_SetVelocityBoundary << < gridDims, blockDims >> >(src);

			for (int i = 0; i < iteration; i++)
			{
				buf.assign(dst);
				K_SetVelocityBoundary << < gridDims, blockDims >> >(buf);
				K_Laplacian << < gridDims, blockDims >> >(dst, src, buf, mass, a);
			}
		}

		__global__ void K_PrepareForProjection(GridCoef coefMatrix, Grid1f RHS, Grid1f mass, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float h, float dt)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = mass.nx();
			int ny = mass.ny();
			int nz = mass.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			float hh = h*h;

			float div_ijk = 0.0f;

			Coef A_ijk;

			A_ijk.a = 0.0f;
			A_ijk.x0 = 0.0f;
			A_ijk.x1 = 0.0f;
			A_ijk.y0 = 0.0f;
			A_ijk.y1 = 0.0f;
			A_ijk.z0 = 0.0f;
			A_ijk.z1 = 0.0f;

			float m_ijk = mass(i, j, k);

			if (i+1 < nx) {
				float c = 0.5f*(m_ijk + mass(i + 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.x1 += term;

// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index + 1, -term);
			}
			div_ijk -= vel_u(i + 1, j, k) / h;
			//rhs[index] -= vel_u(i + 1, j, k) / h;

			//left neighbour
			if (i-1 >= 0) {
				float c = 0.5f*(m_ijk + mass(i - 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.x0 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index - 1, -term);
			}
			div_ijk += vel_u(i, j, k) / h;
			//rhs[index] += vel_u(i, j, k) / h;

			//top neighbour
			if (j+1 < ny) {
				float c = 0.5f*(m_ijk + mass(i, j + 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y1 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index + ni, -term);
			}
			div_ijk -= vel_v(i, j + 1, k) / h;

			//bottom neighbour
			if (j-1 >= 0) {
				float c = 0.5f*(m_ijk + mass(i, j - 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y0 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index - ni, -term);
			}
			div_ijk += vel_v(i, j, k) / h;


			//far neighbour

			if (k+1 < nz) {
				float c = 0.5f*(m_ijk + mass(i, j, k + 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.z1 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index + ni*nj, -term);
			}
			div_ijk -= vel_w(i, j, k + 1) / h;

			//near neighbour

			if (k-1 >= 0) {
				float c = 0.5f*(m_ijk + mass(i, j, k - 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.z0 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index - ni*nj, -term);
			}
			div_ijk += vel_w(i, j, k) / h;

			if (m_ijk > 1.0)
			{
				div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
			}
// 			else if (m_ijk > 0.0f)
// 			{
// 				div_ijk -= 0.001f;
// 			}

			coefMatrix(i, j, k) = A_ijk;
			RHS(i, j, k) = div_ijk;
		}

		void PFKernel::PrepareForProjection(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, GridCoef coefMatrix, Grid1f RHS, Grid1f mass, float h, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(mass.nx(), mass.ny(), mass.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_PrepareForProjection << < gridDims, blockDims >> >(coefMatrix, RHS, mass, vel_u, vel_v, vel_w, h, dt);
		}

		__global__ void K_Projection(Grid1f pressure, Grid1f buf, GridCoef coefMatrix, Grid1f RHS, int numIter)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = pressure.nx();
			int ny = pressure.ny();
			int nz = pressure.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;
			
			int k0 = coefMatrix.index(i, j, k);
			Coef A_ijk = coefMatrix[k0];
			
			float a = A_ijk.a;
			float x0 = A_ijk.x0;
			float x1 = A_ijk.x1;
			float y0 = A_ijk.y0;
			float y1 = A_ijk.y1;
			float z0 = A_ijk.z0;
			float z1 = A_ijk.z1;
//			pressure[k0] = 0.0f;
 			float p_ijk;
// 			for (int it = 0; it < 1; it++)
 			{
// 				buf[k0] = 0.0f;// pressure[k0];
// 				__syncthreads();

				p_ijk = RHS[k0];
				if (i > 0) p_ijk += x0*buf(i - 1, j, k);
				if (i < nx - 1) p_ijk += x1*buf(i + 1, j, k);
				if (j > 0) p_ijk += y0*buf(i, j - 1, k);
				if (j < ny - 1) p_ijk += y1*buf(i, j + 1, k);
				if (k > 0) p_ijk += z0*buf(i, j, k - 1);
				if (k < nz - 1) p_ijk += z1*buf(i, j, k + 1);

				pressure[k0] = p_ijk / a;
			}
		}

		void PFKernel::Projection(Grid1f pressure, Grid1f buf, GridCoef coefMatrix, Grid1f RHS, int numIter)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(RHS.nx(), RHS.ny(), RHS.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			//pressure.reset();
			for (int i = 0; i < numIter; i++)
			{
				//K_CopyData << < gridDims, blockDims >> >(buf, pressure);
				buf.assign(pressure);
				K_Projection << < gridDims, blockDims >> >(pressure, buf, coefMatrix, RHS, numIter);
			}
		}

		__global__ void K_UpdateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f pressure, Grid1f mass, float h, float dt)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = mass.nx();
			int ny = mass.ny();
			int nz = mass.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			if (i == 0) { vel_u(i, j, k) = 0.0f; return; }
			if (i == nx - 1) { vel_u(i + 1, j, k) = 0.0f; return; }
			if (j == 0) { vel_v(i, j, k) = 0.0f; return; }
			if (j == ny - 1) { vel_v(i, j + 1, k) = 0.0f; return; }
			if (k == 0) { vel_w(i, j, k) = 0.0f; return; }
			if (k == nz - 1) { vel_w(i, j, k + 1) = 0.0f; return; }


			int index;
			float c;

			int nxy = nx*ny;

			index = mass.index(i, j, k);
			c = 0.5f*(mass[index - 1] + mass[index]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_u(i, j, k) -= dt*(pressure[index] - pressure[index - 1]) / h / (c*RHO1 + (1.0f - c)*RHO2);

			c = 0.5f*(mass[index] + mass[index - nx]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_v(i, j, k) -= dt*(pressure[index] - pressure[index - nx]) / h / (c*RHO1 + (1.0f - c)*RHO2);

			c = 0.5f*(mass[index] + mass[index - nxy]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_w(i, j, k) -= dt*(pressure[index] - pressure[index - nxy]) / h / (c*RHO1 + (1.0f - c)*RHO2);

		}

		void PFKernel::UpdateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f pressure, Grid1f mass, float h, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(mass.nx(), mass.ny(), mass.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_UpdateVelocity << < gridDims, blockDims >> >(vel_u, vel_v, vel_w, pressure, mass, h, dt);
		}

		__device__ float K_MapDragForce(float v)
		{
			float mag = abs(v);
			float max_weight = 0.9f;
			if (mag > 1.0f) return max_weight;
			else return max_weight*pow(mag, 0.2f);
		}

		__global__ void K_ApplyDragForce(Grid3f vel, float dt)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = vel.nx();
			int ny = vel.ny();
			int nz = vel.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			auto v = vel(i, j, k);
			float weight = K_MapDragForce(v.norm());

//			if (k > nz*0.75f)
//			{
//				weight = 1.0f;
//			}
			vel(i, j, k) = weight*v;
		}

		void PFKernel::ApplyDragForce(Grid3f vel, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(vel.nx(), vel.ny(), vel.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_ApplyDragForce << < gridDims, blockDims >> >(vel, dt);
		}

		__global__ void K_ApplyGravity(
			Grid1f u,
			Grid1f v,
			Grid1f w,
			Vec3f g,
			int nx,
			int ny,
			int nz,
			float dt)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			if(i < nx - 1)
				u(i, j, k) += g.x * dt;

			if(j < ny - 1)
				v(i, j, k) += g.y * dt;

			if(k < nz - 1)
				w(i, j, k) += g.z * dt;
		}

		void PFKernel::ApplyGravity(Grid1f u, Grid1f v, Grid1f w, Vec3f g, 
			int nx,
			int ny,
			int nz, float dt)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(nx, ny, nz);
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_ApplyGravity << < gridDims, blockDims >> > (u, v, w, g, nx, ny, nz, dt);
		}

		__global__ void K_SetU(Grid1f vel_u)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = vel_u.nx();
			int ny = vel_u.ny();
			int nz = vel_u.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			if (i <= 0 || i >= nx-1)
			{
				vel_u(i, j, k) = 0.0f;
			}
		}

		__global__ void K_SetV(Grid1f vel_v)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = vel_v.nx();
			int ny = vel_v.ny();
			int nz = vel_v.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			if (j <= 0 || j >= ny - 1)
			{
				vel_v(i, j, k) = 0.0f;
			}
		}

		__global__ void K_SetW(Grid1f vel_w)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = vel_w.nx();
			int ny = vel_w.ny();
			int nz = vel_w.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			if (k <= 0 || k >= nz - 1)
			{
				vel_w(i, j, k) = 0.0f;
			}

		}

		void PFKernel::SetU(Grid1f vel_u)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(vel_u.nx(), vel_u.ny(), vel_u.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_SetU << < gridDims, blockDims >> >(vel_u);
		}

		void PFKernel::SetV(Grid1f vel_v)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(vel_v.nx(), vel_v.ny(), vel_v.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_SetV << < gridDims, blockDims >> >(vel_v);
		}

		void PFKernel::SetW(Grid1f vel_w)
		{
			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(vel_w.nx(), vel_w.ny(), vel_w.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_SetW << < gridDims, blockDims >> >(vel_w);
		}


		__global__ void K_Reshape(
			DArray<float> mass1d,
			Grid1f mass)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			int k = blockIdx.z * blockDim.z + threadIdx.z;

			int nx = mass.nx();
			int ny = mass.ny();
			int nz = mass.nz();

			if (i >= nx) return;
			if (j >= ny) return;
			if (k >= nz) return;

			mass1d[mass.index(i, j, k)] = mass(i, j, k);
		}

		float PFKernel::calcualteTotalMass(Grid1f mass)
		{
			DArray<float> mass1d(mass.size());

			dim3 gridDims, blockDims;
			uint3 fDims = make_uint3(mass.nx(), mass.ny(), mass.nz());
			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

			K_Reshape << < gridDims, blockDims >> > (mass1d, mass);

			Reduction<float> reduce;
			return reduce.maximum(mass1d.begin(), mass1d.size());
		}

// 		__global__ void k_InputDensity(Grid1f mass, Grid4f pigments, Grid4f pgColor, Grid1f extMass, float* density_brush, float4 * distance_brush, float4* color_brush, float4 fillColor, int3 simOrigin, int3 brushOrigin, int nLayer, float mixRate)
// 		{
// 			int i = blockDim.x * blockIdx.x + threadIdx.x;
// 			int j = blockIdx.y * blockDim.y + threadIdx.y;
// 			int k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx;
// 			int ny = mass.ny;
// 			int nz = mass.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int ix = i + simOrigin.x - brushOrigin.x;
// 			int iy = j + simOrigin.y - brushOrigin.y;
// 			int iz = k + simOrigin.z - brushOrigin.z;
// 
// 			if (ix < 0 || ix >= nx)	return;
// 			if (iy < 0 || iy >= nx)	return;
// 			if (iz < 0 || iz >= nx)	return;
// 
// 			int id = mass.Index(i, j, k);
// 
// 			int bId = mass.Index(ix, iy, iz);
// 			float4 color = color_brush[bId];
// 			float4 dist = distance_brush[bId];
// 			float rho = density_brush[bId];
// 
// 			float m_ijk = mass[id];
// 			float extM = extMass[id];
// 
// 			if (m_ijk < MASS_THESHOLD2 && extM > MASS_THESHOLD2 && rho > 0.01f && k <= nLayer)
// 			{
// 				float ratio = rho;
// 				if (rho > 1.0f)
// 				{
// 					ratio = 1.0f;
// 				}
// 				
// 				mass[id] = 0.6f+(1.0f-ratio)*0.4f;
// 
// 			}
// 
// 			if (rho > 0.01f)
// 			{
// 				float4 oldColor, newColor;
// 				{
// 					oldColor = pigments[id];
// 					newColor = make_float4(color.x, color.y, color.z, 1.0f);
// 					if (oldColor.w > EPSILON)
// 					{
// 						pigments[id] = mixRate*newColor + (1.0f - mixRate)*oldColor;
// 					}
// 					else
// 					{
// 						pigments[id] = newColor;
// 					}
// 					
// 				}
// 
// 				int subId = pgColor.Index(i*MAX_PIGMENTS, j, k);
// 				for (int t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					oldColor = pgColor[subId + t];
// 					newColor = make_float4(color.x, color.y, color.z, 1.0f);
// 					if (oldColor.w > EPSILON)
// 					{
// 						pgColor[subId + t] = mixRate*newColor + (1.0f - mixRate)*oldColor;
// 					}
// 					else
// 					{
// 						pgColor[subId + t] = newColor;
// 					}
// 				}
// 			}
// 			
// 		}
// 
// 		void PFKernel::InputDensity(Grid1f mass, Grid1f mb0, Grid1f mb1, Grid4f pigments, Grid4f cb0, Grid4f cb1, Grid4f subPigs, uint3 cvOrigin, float dt)
// 		{
// 			float4* color_brush_field = (float4*)MEM_Brush.ColorField;
// 			float4* distance_brush_field = (float4*)MEM_Brush.DistanceField;
// 			float* density_brush_field = (float*)MEM_Brush.DensityField;
// 
// 			int nLayer = (int)8.0f*PARAM_Brush.Wetness;
// 			nLayer = clamp(nLayer, 0, 8);
// 
// 			float mixRate = PARAM_FLIPFluid.CanvasMixingRate;
// 
// 			int3 simOrigin = make_int3(cvOrigin.x, cvOrigin.y, cvOrigin.z);
// 			int3 brushOrigin = make_int3(PARAM_Brush.Field_GridOrigin.x, PARAM_Brush.Field_GridOrigin.y, PARAM_Brush.Field_GridOrigin.z);
// 
// 			mb0.CopyFrom(mass);
// 			mb1.CopyFrom(mass);
// 			cb0.CopyFrom(pigments);
// 			cb1.CopyFrom(pigments);
// 			PFKernel::ExtraploateMass(cb0, cb1, mb0, mb1, density_brush_field, simOrigin, brushOrigin, nLayer);
// 
// 			LAUNCH_KERNEL(k_InputDensity, make_uint3(mass.nx, mass.ny, mass.nz), make_uint3(32, 8, 1), mass, pigments, subPigs, mb0, density_brush_field, distance_brush_field, color_brush_field, PARAM_Brush.RefillPigment, simOrigin, brushOrigin, nLayer, mixRate);
// 		}

// 		__global__ void K_InputVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, uint3 originNow, float3 center, float3 vel)
// 		{
// 			int i = blockDim.x * blockIdx.x + threadIdx.x;
// 			int j = blockIdx.y * blockDim.y + threadIdx.y;
// 			int k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = vel_v.nx();
// 			int ny = vel_w.ny();
// 			int nz = vel_u.nz();
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// // 			float mag_v = length(vel);
// // 			if (mag_v > EPSILON)
// // 			{
// // 				vel = normalize(vel);
// // 			}
// 			
// 			float3 p, dir;
// 			float dist;
// 
// 			center -= make_float3(originNow);
// 
// 			float rad = 10.0f;
// 
// 			float mag_v = length(vel);
// 
// 			if (mag_v > 5.0f)
// 			{
// 				vel *= 5.0f / mag_v;
// 			}
// 
// 			p.x = i - 0.5f;	p.y = j;	p.z = k;
// 			dir = p - center;
// 			dist = length(dir);
// 
// 			if (dist < rad)
// 			{
// 				float weight = 8.0f*(rad - dist) / float(rad);
// 				if (weight > 1.0f) weight = 1.0f;
// 				
// 				vel_u(i, j, k) = weight*vel.x;// *(rad - dist) / rad;
// 			}
// 
// 			p.x = i;	p.y = j - 0.5f;	p.z = k;
// 			dir = p - center;
// 			dist = length(dir);
// 			if (dist < rad)
// 			{
// 				float weight = 8.0f*(rad - dist) / float(rad);
// 				if (weight > 1.0f) weight = 1.0f;
// 
// 				vel_v(i, j, k) = weight*vel.y;// *(rad - dist) / rad;;
// 			}
// 
// 			p.x = i;	p.y = j;	p.z = k - 0.5f;
// 			dir = p - center;
// 			dist = length(dir);
// 			if (dist < rad)
// 			{
// 				float weight = 8.0f*(rad - dist) / float(rad);
// 				if (weight > 1.0f) weight = 1.0f;
// 
// 				if (vel.z > 0.0f)
// 				{
// 					vel_w(i, j, k) = 0.0f;
// 				}
// 				else
// 				{
// 					vel_w(i, j, k) = weight*vel.z;
// 				}
// 				
// 			}
// 		}
// 
// 
// 		__global__	void k_InputBrush(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f mass, float * density_brush, float3 * velocity_brush, int3 simOrigin, int3 brushOrigin)
// 		{
// 			int i = blockDim.x * blockIdx.x + threadIdx.x;
// 			int j = blockIdx.y * blockDim.y + threadIdx.y;
// 			int k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx();
// 			int ny = mass.ny();
// 			int nz = mass.nz();
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int ix = i + simOrigin.x - brushOrigin.x;
// 			int iy = j + simOrigin.y - brushOrigin.y;
// 			int iz = k + simOrigin.z - brushOrigin.z;
// 			
// 			if (ix < 0 || ix >= nx)	return;
// 			if (iy < 0 || iy >= nx)	return;
// 			if (iz < 0 || iz >= nx)	return;
// 
// 			int id = mass.index(i, j, k);
// 
// 			int bId = mass.index(ix, iy, iz);
// 
// 			float m_ijk = density_brush[bId];
// 			if (m_ijk > 0.01f)
// 			{
// 				float3 v = velocity_brush[bId];
// 				vel_u(i, j, k) = v.x / 100.0f;
// 				vel_v(i, j, k) = v.y / 100.0f;
// 				vel_w(i, j, k) = v.z / 100.0f;
// 				if (v.z > 0.0f)
// 				{
// 					vel_w(i, j, k) = 0.0f;
// 				}
// 				if (k < 1)
// 				{
// 					vel_u(i, j, k) = 0.0f;
// 					vel_v(i, j, k) = 0.0f;
// 					vel_w(i, j, k) = 0.0f;
// 				}
// 			}

// 			float3 cellCenter_pos = make_float3(index_x + 0.5f, index_y + 0.5f, index_z + 0.5f) * d_manager_params.flipfluid.GridCellSize + d_manager_params.flipfluid.GridOrigin;
// 
// 			{
// 				float3 cellcenter_pos = make_float3(index_x, index_y + 0.5f, index_z + 0.5f) * d_manager_params.flipfluid.GridCellSize + d_manager_params.flipfluid.GridOrigin;
// 				float3 cellpos_brush = (cellcenter_pos - d_manager_params.brush.Field_GridOrigin) / d_manager_params.brush.Field_GridCellSize;
// 
// 				uint index_x_canvas = index_x + d_manager_params.flipfluid.GridOrigin_i.x;
// 				uint index_y_canvas = index_y + d_manager_params.flipfluid.GridOrigin_i.y;
// 				uint index_z_canvas = index_z + d_manager_params.flipfluid.GridOrigin_i.z;
// 				uint index_canvas = kd_cell_index(index_x_canvas, index_y_canvas, index_z_canvas, d_manager_params.common.CanvasDims);
// 				if (LevelSetDry[index_canvas] >= 0.f)
// 				{
// 					if (cellpos_brush.x < nx_brush - 1 && cellpos_brush.x>1
// 						&& cellpos_brush.y < ny_brush - 1 && cellpos_brush.y>1
// 						&& cellpos_brush.z > 1)
// 					{
// 						cellpos_brush.z = clamp(cellpos_brush.z, 1.f, nz_brush - 1.f);
// 						float density_brush_this = kd_bilinear_sample_cellposf(density_brush, d_manager_params.brush.Field_GridDims, cellpos_brush);
// 
// 						if (density_brush_this > 0.001)
// 						{
// 							float3 velocity_brush_this = kd_bilinear_sample_cellposf(velocity_brush, d_manager_params.brush.Field_GridDims, cellpos_brush);
// 							velocity_fluid[index_fluid] = mixRate * velocity_brush_this / dissipationPerStep + (1.0f - mixRate) * velocity_fluid[index_fluid];
// 						}
// 					}
// 				}
// 				else
// 				{
// 					velocity_fluid[index_fluid] = make_float3(0.0f, 0.0f, 0.0f);
// 				}
// 			}
//		}

		int simItor = 0;

// 		void PFKernel::InputVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f mass, uint3 cvOrigin, float dt)
// 		{
// 			if (PARAM_Brush.IsDrawing)
// 			{
// #ifdef INPUT_BALL
// 
// 				dim3 gridDims, blockDims;
// 				uint3 fDims = make_uint3(vel_v.nx, vel_w.ny, vel_u.nz);
// 				computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 				float3 brushPos, oldPos, vel;
// // 				if (simItor < 128)
// // 				{
// // 					brushPos = make_float3((float)simItor, 64.0f, 18.0f);
// // 					vel = 2.0f*make_float3(1.0f, 0.0f, 0.0f);
// // 				}
// // 				else
// 				{
// 					brushPos = PARAM_Brush.BrushPos;
// 					oldPos = PARAM_Brush.BrushPos_last;
// 					vel = (brushPos - oldPos) / dt;
// 				}
// 				std::cout << brushPos.x << " " << brushPos.y << " " << brushPos.z << std::endl;
// 
// 				K_InputVelocity << < gridDims, blockDims >> >(vel_u, vel_v, vel_w, cvOrigin, brushPos, vel);
// #endif
// 
// #ifdef INPUT_BRUSH
// 				float3* velocity_brush_field = (float3*)MEM_Brush.VelocityField;
// 				float* density_brush_field = (float*)MEM_Brush.DensityField;
// 				float4* color_brush_field = (float4*)MEM_Brush.ColorField;
// 				float* distance_brush_field = (float*)MEM_Brush.DistanceField;
// 
// 				int3 simOrigin = make_int3(cvOrigin.x, cvOrigin.y, cvOrigin.z);
// 				int3 brushOrigin = make_int3(PARAM_Brush.Field_GridOrigin.x, PARAM_Brush.Field_GridOrigin.y, PARAM_Brush.Field_GridOrigin.z);
// 
// 				LAUNCH_KERNEL(k_InputBrush, make_uint3(mass.nx, mass.ny, mass.nz), make_uint3(32, 8, 1), vel_u, vel_v, vel_w, mass, density_brush_field, velocity_brush_field, simOrigin, brushOrigin);
// 				
// // 				std::cout << "Field Size: " << PARAM_Brush.Field_GridDims.x << " " << PARAM_Brush.Field_GridDims.y << " " << PARAM_Brush.Field_GridDims.z << std::endl;
// // 				std::cout << "origin Size: " << PARAM_Brush.Field_GridOrigin.x << " " << PARAM_Brush.Field_GridOrigin.y << " " << PARAM_Brush.Field_GridOrigin.z << std::endl;
// 
// 
// 
// #endif
// 
// 				simItor++;
// 			}
// 		}

// 		__global__ void K_DrawSetup(cudaSurfaceObject_t surf_paintdensity, cudaSurfaceObject_t surf_pigment, Grid1f density, Grid4f pigment, uint3 originNow)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = density.nx;
// 			int ny = density.ny;
// 			int nz = density.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			if (k > nz*0.75f)
// 			{
// 				density(i, j, k) = 0.0f;
// 			}
// 
// 			float density_towrite = density(i, j, k);
// 			float4 color = pigment(i, j, k);
// 			density_towrite = density_towrite > 1.0f ? 1.0f : density_towrite;
// 			density_towrite = density_towrite < 0.0f ? 0.0f : density_towrite;
// 			float4 pigment_towrite = make_float4(color.x, color.y, color.z, color.w);
// 
// 			int i_cv = i + originNow.x;
// 			int j_cv = j + originNow.y;
// 			int k_cv = k + originNow.z;
// 
// 			int minX = min(i, nx-1-i);
// 			int minY = min(j, ny-1-j);
// 			int minDist = min(minX, minY);
// 
// 			int width = 20;
// 
// 			if (minDist > width)
// 			{
// 				minDist = width;
// 			}
// 
// 			float w = minDist / (float)width;
// 			
// 			float cv_density;
// 			surf3Dread(&cv_density, surf_paintdensity, i_cv*sizeof(float), j_cv, k_cv);
// 
// 
// 			density_towrite = w*density_towrite + (1.0f - w)*cv_density;
// 
// 			surf3Dwrite(density_towrite, surf_paintdensity, i_cv*sizeof(float), j_cv, k_cv);
// 			surf3Dwrite(pigment_towrite, surf_pigment, i_cv*sizeof(float4), j_cv, k_cv);
// 		}
// 
// 		void PFKernel::SetupDensityField(Grid1f field, Grid4f pigment, uint3 cvOrigin)
// 		{
// 			cudaSurfaceObject_t PaintDensity_SurfaceObj = 0;
// 			cudaSurfaceObject_t Pigment_SurfaceObj = 0;
// 
// 			PaintDensity_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.PaintDensity_cudaArray);
// 			Pigment_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.Pigment_cudaArray);
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(field.nx, field.ny, field.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			K_DrawSetup << < gridDims, blockDims >> >(PaintDensity_SurfaceObj, Pigment_SurfaceObj, field, pigment, cvOrigin);
// 
// 			checkCudaErrors(cudaDestroySurfaceObject(PaintDensity_SurfaceObj));
// 			checkCudaErrors(cudaDestroySurfaceObject(Pigment_SurfaceObj));
// 			// 
// 			// 		K_MPMDrawSetup <<<gridDim, blockDim >>>(buffer, particles);
// 		}

// 		__global__ void K_SetupParticles(float3* pos, float4* color, Grid3f gPos, Grid4f gColor)
// 		{
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = gPos.nx/MAX_PIGMENTS;
// 			int ny = gPos.ny;
// 			int nz = gPos.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int id = gPos.Index(i*MAX_PIGMENTS, j, k);
// 
// 			//for (int t = 0; t < MAX_PIGMENTS; t++)
// 			{
// 				pos[id] = gPos[id];
// 				color[id] = gColor[id];
// 			}
// 		}
// 
// 		void PFKernel::SetupParticles(Grid3f gPos, Grid4f gColor)
// 		{
// 			PARAM_FLIPFluid.NumParticles = gColor.Size()/MAX_PIGMENTS;
// 			float3* posBuf = (float3 *)MEM_FLIPFluid.ParticlePosition[MEM_FLIPFluid.ppIndex.particles];
// 			float4* colorBuf = (float4 *)MEM_FLIPFluid.ParticleColor[MEM_FLIPFluid.ppIndex.particles];
// 
// 			LAUNCH_KERNEL(K_SetupParticles, make_uint3(gColor.nx / MAX_PIGMENTS, gColor.ny, gColor.nz), make_uint3(32, 8, 1), posBuf, colorBuf, gPos, gColor);
// 		}

// 		__global__ void K_InitialCanvas(cudaSurfaceObject_t surf_paintdensity, cudaSurfaceObject_t surf_pigment, uint3 cvSize)
// 		{
// 
// 			uint i = blockDim.x * blockIdx.x + threadIdx.x;
// 			uint j = blockIdx.y * blockDim.y + threadIdx.y;
// 			uint k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = cvSize.x;
// 			int ny = cvSize.y;
// 			int nz = cvSize.z;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			float density_towrite;
// 			if (k < cvSize.z / 2)
// 			{
// 				density_towrite = 0.0f;
// 			}
// 			else
// 				density_towrite = 0.0f;
// 
// 			float4 pigment_towrite;
// 			if (i % 20 < 10/* && j % 20 < 10*/)
// 			{
// 				pigment_towrite = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
// 			}
// 			else
// 			{
// 				pigment_towrite = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
// 			}
//  
// 			surf3Dwrite(density_towrite, surf_paintdensity, i*sizeof(float), j, k);
// 			surf3Dwrite(pigment_towrite, surf_pigment, i*sizeof(float4), j, k);
// 		}
// 
// 
// 		void PFKernel::InitialCanvas()
// 		{
// 			cudaSurfaceObject_t PaintDensity_SurfaceObj = 0;
// 			cudaSurfaceObject_t Pigment_SurfaceObj = 0;
// 
// 			PaintDensity_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.PaintDensity_cudaArray);
// 			Pigment_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.Pigment_cudaArray);
// 
// 			uint3 cvSize = PARAM_COMMON.CanvasDims;
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(cvSize.x, cvSize.y, cvSize.z);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			K_InitialCanvas << < gridDims, blockDims >> >(PaintDensity_SurfaceObj, Pigment_SurfaceObj, cvSize);
// 
// 			checkCudaErrors(cudaDestroySurfaceObject(PaintDensity_SurfaceObj));
// 			checkCudaErrors(cudaDestroySurfaceObject(Pigment_SurfaceObj));
// 		}

// 		__global__ void K_MoveSimWindow(Grid1f mass, Grid1f pre_mass, Grid4f pigments, Grid4f pre_pigments, Grid1f vel_u, Grid1f pre_vel_u, Grid1f vel_v, Grid1f pre_vel_v, Grid1f vel_w, Grid1f pre_vel_w, cudaSurfaceObject_t surf_paintdensity, cudaSurfaceObject_t surf_pigment, uint3 originLast, uint3 originNow)
// 		{
// 			int i = blockDim.x * blockIdx.x + threadIdx.x;
// 			int j = blockIdx.y * blockDim.y + threadIdx.y;
// 			int k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx;
// 			int ny = mass.ny;
// 			int nz = mass.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int i_cv = originNow.x + i;
// 			int j_cv = originNow.y + j;
// 			int k_cv = originNow.z + k;
// 
// 			int i_last = i + originNow.x - originLast.x;
// 			int j_last = j + originNow.y - originLast.y;
// 			int k_last = k + originNow.z - originLast.z;
// 
// 			if (i_last >= 0 && i_last < nx && j_last >= 0 && j_last < ny && k_last >= 0 && k_last < nz)
// 			{
// 				mass(i, j, k) = pre_mass(i_last, j_last, k_last);
// 				pigments(i, j, k) = pre_pigments(i_last, j_last, k_last);
// 
// 				vel_u(i, j, k) = pre_vel_u(i_last, j_last, k_last);
// 				vel_v(i, j, k) = pre_vel_v(i_last, j_last, k_last);
// 				vel_w(i, j, k) = pre_vel_w(i_last, j_last, k_last);
// 			}
// 			else
// 			{
// 				float density_i;
// 				float4 pigment_i;
// 				surf3Dread(&density_i, surf_paintdensity, i_cv*sizeof(float), j_cv, k_cv);
// 				surf3Dread(&pigment_i, surf_pigment, i_cv*sizeof(float4), j_cv, k_cv);
// 
// 				mass(i, j, k) = density_i;
// 				pigments(i, j, k) = pigment_i;
// 			}
// 		}

// 		__global__ void K_MoveSimWindow(Grid1f mass, Grid1f pre_mass, Grid1f vel_u, Grid1f pre_vel_u, Grid1f vel_v, Grid1f pre_vel_v, Grid1f vel_w, Grid1f pre_vel_w, cudaSurfaceObject_t surf_paintdensity, cudaSurfaceObject_t surf_pigment, uint3 originLast, uint3 originNow)
// 		{
// 			int i = blockDim.x * blockIdx.x + threadIdx.x;
// 			int j = blockIdx.y * blockDim.y + threadIdx.y;
// 			int k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = mass.nx;
// 			int ny = mass.ny;
// 			int nz = mass.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int i_cv = originNow.x + i;
// 			int j_cv = originNow.y + j;
// 			int k_cv = originNow.z + k;
// 
// 			int i_last = i + originNow.x - originLast.x;
// 			int j_last = j + originNow.y - originLast.y;
// 			int k_last = k + originNow.z - originLast.z;
// 
// 			int id = mass.Index(i, j, k);
// 
// 			if (i_last >= 0 && i_last < nx && j_last >= 0 && j_last < ny && k_last >= 0 && k_last < nz)
// 			{
// 				int id_last = pre_mass.Index(i_last, j_last, k_last);
// 				mass[id] = pre_mass[id_last];
// 				//pigments(i, j, k) = pre_pigments(i_last, j_last, k_last);
// 
// 				vel_u(i, j, k) = pre_vel_u(i_last, j_last, k_last);
// 				vel_v(i, j, k) = pre_vel_v(i_last, j_last, k_last);
// 				vel_w(i, j, k) = pre_vel_w(i_last, j_last, k_last);
// 			}
// 			else
// 			{
// 				float density_i;
// 				float4 pigment_i;
// 				surf3Dread(&density_i, surf_paintdensity, i_cv*sizeof(float), j_cv, k_cv);
// 				surf3Dread(&pigment_i, surf_pigment, i_cv*sizeof(float4), j_cv, k_cv);
// 
// 				mass[id] = density_i;
// 
// 				//pigments(i, j, k) = pigment_i;
// 			}
// 		}


// 		void PFKernel::MoveSimWindow(Grid1f mass, Grid1f pre_mass, Grid4f pigments, Grid4f pre_pigments, Grid1f vel_u, Grid1f pre_vel_u, Grid1f vel_v, Grid1f pre_vel_v, Grid1f vel_w, Grid1f pre_vel_w, uint3 originLast, uint3 originNow)
// 		{
// 			cudaSurfaceObject_t PaintDensity_SurfaceObj = 0;
// 			cudaSurfaceObject_t Pigment_SurfaceObj = 0;
// 
// 			PaintDensity_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.PaintDensity_cudaArray);
// 			Pigment_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.Pigment_cudaArray);
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(mass.nx, mass.ny, mass.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			mass.Clear();
// 			pigments.Clear();
// 			vel_u.Clear();
// 			vel_v.Clear();
// 			vel_w.Clear();
// 
// 			K_MoveSimWindow << <gridDims, blockDims >> >(mass, pre_mass, pigments, pre_pigments, vel_u, pre_vel_u, vel_v, pre_vel_v, vel_w, pre_vel_w, PaintDensity_SurfaceObj, Pigment_SurfaceObj, originLast, originNow);
// 			
// 			checkCudaErrors(cudaDestroySurfaceObject(PaintDensity_SurfaceObj));
// 			checkCudaErrors(cudaDestroySurfaceObject(Pigment_SurfaceObj));
// 		}

// 		void PFKernel::MoveSimWindow(Grid1f mass, Grid1f pre_mass, Grid1f vel_u, Grid1f pre_vel_u, Grid1f vel_v, Grid1f pre_vel_v, Grid1f vel_w, Grid1f pre_vel_w, uint3 originLast, uint3 originNow)
// 		{
// 			cudaSurfaceObject_t PaintDensity_SurfaceObj = 0;
// 			cudaSurfaceObject_t Pigment_SurfaceObj = 0;
// 
// 			PaintDensity_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.PaintDensity_cudaArray);
// 			Pigment_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.Pigment_cudaArray);
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(mass.nx, mass.ny, mass.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			mass.Clear();
// 			vel_u.Clear();
// 			vel_v.Clear();
// 			vel_w.Clear();
// 
// 			K_MoveSimWindow << <gridDims, blockDims >> >(mass, pre_mass, vel_u, pre_vel_u, vel_v, pre_vel_v, vel_w, pre_vel_w, PaintDensity_SurfaceObj, Pigment_SurfaceObj, originLast, originNow);
// 
// 			checkCudaErrors(cudaDestroySurfaceObject(PaintDensity_SurfaceObj));
// 			checkCudaErrors(cudaDestroySurfaceObject(Pigment_SurfaceObj));
// 		}


// 		__global__ void K_MoveSimWindow(Grid4f color, Grid4f preColor, Grid3f pos, Grid3f prePos, Grid1i num, Grid1i preNum, Grid1f acc, Grid1f preAcc, cudaSurfaceObject_t surf_paintdensity, cudaSurfaceObject_t surf_pigment, uint3 originLast, uint3 originNow)
// 		{
// 			int i = blockDim.x * blockIdx.x + threadIdx.x;
// 			int j = blockIdx.y * blockDim.y + threadIdx.y;
// 			int k = blockIdx.z * blockDim.z + threadIdx.z;
// 
// 			int nx = color.nx/MAX_PIGMENTS;
// 			int ny = color.ny;
// 			int nz = color.nz;
// 
// 			if (i >= nx) return;
// 			if (j >= ny) return;
// 			if (k >= nz) return;
// 
// 			int i_cv = originNow.x + i;
// 			int j_cv = originNow.y + j;
// 			int k_cv = originNow.z + k;
// 
// 			int i_last = i + originNow.x - originLast.x;
// 			int j_last = j + originNow.y - originLast.y;
// 			int k_last = k + originNow.z - originLast.z;
// 
// 			int pId = color.Index(i, j, k);
// 
// 			int id = color.Index(i*MAX_PIGMENTS, j, k);
// 
// 			if (i_last >= 0 && i_last < nx && j_last >= 0 && j_last < ny && k_last >= 0 && k_last < nz)
// 			{
// 				int id_last = preColor.Index(i_last*MAX_PIGMENTS, j_last, k_last);
// 				for (int t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					color[id+t] = preColor[id_last+t];
// 					pos[id + t] = prePos[id_last + t] - make_float3(originNow.x - originLast.x, originNow.y - originLast.y, originNow.z - originLast.z);
// 					num[id + t] = preNum[id_last + t];
// 					acc[id + t] = preAcc[id_last + t];
// 				}
// 			}
// 			else
// 			{
// 				float4 pigment_i;
// 				surf3Dread(&pigment_i, surf_pigment, i_cv*sizeof(float4), j_cv, k_cv);
// 
// 				for (int t = 0; t < MAX_PIGMENTS; t++)
// 				{
// 					color[id + t] = pigment_i;
// 					float3 subPos;
// 					SeedPigment(subPos, t, MAX_PIGMENTS);
// 					pos[id + t] = subPos + make_float3(i, j, k);
// 					num[id + t] = 1;
// 					acc[id + t] = 1.0f;
// 				}
// 			}
// 		}

// 		void PFKernel::MovePigments(Grid4f color, Grid4f preColor, Grid3f pos, Grid3f prePos, Grid1i num, Grid1i preNum, Grid1f acc, Grid1f preAcc, uint3 originLast, uint3 originNow)
// 		{
// 			cudaSurfaceObject_t PaintDensity_SurfaceObj = 0;
// 			cudaSurfaceObject_t Pigment_SurfaceObj = 0;
// 
// 			PaintDensity_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.PaintDensity_cudaArray);
// 			Pigment_SurfaceObj = CUDAMapCudaArrayToSurfaceObject(MEM_FLIPFluid.Pigment_cudaArray);
// 
// 			dim3 gridDims, blockDims;
// 			uint3 fDims = make_uint3(preColor.nx/MAX_PIGMENTS, preColor.ny, preColor.nz);
// 			computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
// 
// 			K_MoveSimWindow << <gridDims, blockDims >> >(color, preColor, pos, prePos, num, preNum, acc, preAcc, PaintDensity_SurfaceObj, Pigment_SurfaceObj, originLast, originNow);
// 
// 			checkCudaErrors(cudaDestroySurfaceObject(PaintDensity_SurfaceObj));
// 			checkCudaErrors(cudaDestroySurfaceObject(Pigment_SurfaceObj));
// 		}

// 		void PFKernel::Step()
// 		{
// 
// 		}

// 		void PFKernel::SetPhaseFieldParameters(PFParameter* params)
// 		{
// 			checkCudaErrors(cudaMemcpyToSymbol(pfParams, params, sizeof(PFParameter), 0, cudaMemcpyHostToDevice));
// 		}

}