#include "device.hpp"
#include "texture_binder.hpp"
#include "../internal.hpp"
#include <stdio.h>

using namespace kfusion::device;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume initialization

namespace kfusion
{
    namespace device
    {
        __global__ void clear_volume_kernel(ColorVolume color)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < color.dims.x && y < color.dims.y)
            {
                uchar4 *beg = color.beg(x, y);
                uchar4 *end = beg + color.dims.x * color.dims.y * color.dims.z;

                for(uchar4* pos = beg; pos != end; pos = color.zstep(pos))
                    *pos = make_uchar4 (0, 0, 0, 0);
            }
        }
    }
}

void kfusion::device::clear_volume(ColorVolume volume)
{
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (volume.dims.x, block.x);
    grid.y = divUp (volume.dims.y, block.y);

    clear_volume_kernel<<<grid, block>>>(volume);
    cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume integration

namespace kfusion
{
    namespace device
    {
        texture<uchar4, 2> image_tex(0, cudaFilterModePoint, cudaAddressModeBorder);
        texture<float, 2> depth_tex(0, cudaFilterModePoint, cudaAddressModeBorder, cudaCreateChannelDescHalf());

        struct ColorIntegrator {
            Aff3f vol2cam;
            PtrStep<float> vmap;
            Projector proj;
            int2 im_size;

            float tranc_dist_inv;

            __kf_device__
            void operator()(ColorVolume& volume) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume.dims.x || y >= volume.dims.y)
                    return;

                float3 zstep = make_float3(vol2cam.R.data[0].z, vol2cam.R.data[1].z, vol2cam.R.data[2].z) * volume.voxel_size.z;

                float3 vx = make_float3(x * volume.voxel_size.x, y * volume.voxel_size.y, 0);
                float3 vc = vol2cam * vx; //tranform from volume coo frame to camera one

                ColorVolume::elem_type* vptr = volume.beg(x, y);
                for(int i = 0; i < volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
                {
                    float2 coo = proj(vc); // project to image coordinate
                    // check wether coo in inside the image boundaries
                    if (coo.x >= 0.0 && coo.y >= 0.0 &&
                        coo.x < im_size.x && coo.y < im_size.y) {

                        float Dp = tex2D(depth_tex, coo.x, coo.y);
                        if(Dp == 0 || vc.z <= 0)
                            continue;

                        bool update = false;
                        // Check the distance
                        float sdf = Dp - __fsqrt_rn(dot(vc, vc)); //Dp - norm(v)
                        update = sdf > -volume.trunc_dist && sdf < volume.trunc_dist;
                        if (update)
                        {
                            // Read the existing value and weight
                            uchar4 volume_rgbw = *vptr;
                            uchar weight_prev = volume_rgbw.w;

                            // Average with new value and weight
                            uchar4 rgb = tex2D(image_tex, coo.x, coo.y);

                            uchar Wrk = 1;
                            uchar new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
                            uchar new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
                            uchar new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);

                            uchar weight_new = min(weight_prev + 1, 255);

                            uchar4 volume_rgbw_new;
                            volume_rgbw_new.x = min(255, new_x);
                            volume_rgbw_new.y = min(255, new_y);
                            volume_rgbw_new.z = min(255, new_z);
                            volume_rgbw_new.w = min(volume.max_weight, weight_new);

                            // Write back
                            *vptr = volume_rgbw_new;
                        }
                    } // in camera image range
                } // for (int i=0; i<volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
            } // void operator()
        };

        __global__ void integrate_kernel(const ColorIntegrator integrator, ColorVolume volume) {integrator(volume);};
    }
}

void kfusion::device::integrate(const PtrStepSz<uchar4>& rgb_image,
                                const PtrStepSz<ushort>& depth_map,
                                ColorVolume& volume,
                                const Aff3f& aff,
                                const Projector& proj)
{
    ColorIntegrator ti;
    ti.im_size = make_int2(rgb_image.cols, rgb_image.rows);
    ti.vol2cam = aff;
    ti.proj = proj;
    ti.tranc_dist_inv = 1.f/volume.trunc_dist;

    image_tex.filterMode = cudaFilterModePoint;
    image_tex.addressMode[0] = cudaAddressModeBorder;
    image_tex.addressMode[1] = cudaAddressModeBorder;
    image_tex.addressMode[2] = cudaAddressModeBorder;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    TextureBinder image_binder(rgb_image, image_tex, channelDesc); (void)image_binder;

    depth_tex.filterMode = cudaFilterModePoint;
    depth_tex.addressMode[0] = cudaAddressModeBorder;
    depth_tex.addressMode[1] = cudaAddressModeBorder;
    depth_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder depth_binder(depth_map, depth_tex, cudaCreateChannelDescHalf()); (void)depth_binder;

    dim3 block(32, 8);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    integrate_kernel<<<grid, block>>>(ti, volume);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall ( cudaDeviceSynchronize() );
}

namespace kfusion
{
    namespace device
    {
        struct ColorFetcher
        {
            ColorVolume volume;

            float3 cell_size;
            int n_pts;
            const float4* pts_data;
            Aff3f aff_inv;

            ColorFetcher(const ColorVolume& volume, float3 cell_size);

            __kf_device__
            void operator()(PtrSz<Color> colors) const
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < n_pts)
                {
                  int3 v;
                  float4 p = *(const float4 *) (pts_data + idx);
                  float3 px = make_float3(p.x, p.y, p.z);
                  float3 pv = aff_inv * px;
                  v.x = __float2int_rd(pv.x / cell_size.x);        // round to negative infinity
                  v.y = __float2int_rd(pv.y / cell_size.y);
                  v.z = __float2int_rd(pv.z / cell_size.z);

                  uchar4 rgbw = *volume(v.x, v.y, v.z);
                  uchar4 *pix = colors.data;
                  pix[idx] = rgbw; //bgra
                }
            }
        };

        inline ColorFetcher::ColorFetcher(const ColorVolume& _volume, float3 _cell_size)
        : volume(_volume), cell_size(_cell_size) {}

        __global__ void fetchColors_kernel (const ColorFetcher colorfetcher, PtrSz<Color> colors)
        {colorfetcher(colors);};
    }
}

void
kfusion::device::fetchColors(const ColorVolume& volume, const Aff3f& aff_inv, const PtrSz<Point>& points, PtrSz<Color>& colors)
{
    const int block = 256;

    if (points.size != colors.size || points.size == 0)
        return;

    float3 cell_size = make_float3 (volume.voxel_size.x, volume.voxel_size.y, volume.voxel_size.z);

    ColorFetcher cf(volume, cell_size);
    cf.n_pts = points.size;
    cf.pts_data = points.data;
    cf.aff_inv = aff_inv;

    fetchColors_kernel<<<divUp (points.size, block), block>>>(cf, colors);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
};