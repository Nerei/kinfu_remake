//
// Created by gabriel on 29.06.16.
//

#ifndef KFUSION_COLOR_VOLUME_HPP
#define KFUSION_COLOR_VOLUME_HPP

#include <kfusion/types.hpp>

namespace kfusion
{
    namespace cuda
    {
        /**
         * @class ColorVolume
         * @brief Color volume class
         */
        class KF_EXPORTS ColorVolume
        {
        public:

#pragma mark -
#pragma mark Initialization

            /**
             * @name ColorVolume
             * @fn ColorVolume (const cv::Vec3i &dims)
             * @brief Constructor of the class
             * @param[in] Number of voxels for each dimensions
             */
            ColorVolume (const cv::Vec3i &dims);

            /**
             * @name ~ColorVolume
             * @fn virtual ~ColorVolume(void)
             * @brief Destructor of the class
             */
            virtual ~ColorVolume (void);

            /**
             * @name create
             * @fn void create(const Vec3i& dims)
             * @brief Initialize the volume on the device
             * @param[in] Number of voxels for each dimensions
             */
            void create(const Vec3i& dims);

#pragma mark -
#pragma mark Getters and Setters

            /**
             * @name getDims
             * @fn Vec3i getDims() const
             * @brief Getter for the dimensions
             * @return  Number of voxels along each dimensions
             */
            Vec3i getDims() const {return dims_;}

            /**
             * @name getVoxelSize
             * @fn Vec3f getVoxelSize() const
             * @brief Size of a voxel, given by each dimension's size divided by
             *        the number of voxel along that dimension
             * @return Vector of voxel size along each dimension
             */
            Vec3f getVoxelSize() const {return Vec3f(size_[0]/dims_[0], size_[1]/dims_[1], size_[2]/dims_[2]);}

            /**
             * @name data
             * @fn const CudaData data() const
             * @brief Const getter for the data
             */
            const CudaData data() const {return data_;}

            /**
             * @name data
             * @fn const CudaData data() const
             * @brief Non-const getter for the data
             */
            CudaData data() {return data_;}

            /**
             * @name getSize
             * @fn Vec3f getSize() const
             * @brief Getter for volume size
             */
            Vec3f getSize() const {return size_;}

            /**
             * @name setSize
             * @fn void setSize(const Vec3f& size)
             * @brief Setter for volume size
             */
            void setSize(const Vec3f& size) {size_ = size; setTruncDist(trunc_dist_);}

            /**
             * @name getTruncDist
             * @fn float getTruncDist() const
             * @brief Getter for the TruncDist
             */
            float getTruncDist() const {return trunc_dist_;}

            /**
             * @name setTruncDist
             * @fn void setTruncDist(float distance)
             * @brief Setter for the TruncDist
             */
            void setTruncDist(float distance);

            /**
             * @name getMaxWeight
             * @fn int getMaxWeight() const
             * @brief Getter for the MaxWeight
             */
            int getMaxWeight() const {return max_weight_;}

            /**
             * @name setMaxWeight
             * @fn void getMaxWeight(int weight)
             * @brief Setter for the MaxWeight
             */
            void setMaxWeight(int weight) {max_weight_ = weight;}

            /**
             * @name getPose
             * @fn Affine3f getPose() const
             * @brief Getter for the pose
             */
            Affine3f getPose() const {return pose_;}

            /**
             * @name setPose
             * @fn void setPose(const Affine3f& pose)
             * @brief Setter for the pose
             */
            void setPose(const Affine3f& pose) {pose_ = pose;}

#pragma mark -
#pragma mark Usage

            /**
             * @name clear
             * @fn virtual void clear()
             * @brief Allocate memory on device and initialize at 0
             */
            virtual void clear();

            /**
             * @name applyAffine
             * @fn virtual void applyAffine(const Affine3f& affine)
             * @brief Apply an affine transformation on pose
             * @param[in] affine, the transformation to apply
             */
            virtual void applyAffine(const Affine3f& affine) {pose_ = affine * pose_;}

            /**
             * @name integrate
             * @fn virtual void integrate(const Image& rgb_image, const Affine3f& camera_pose, const Intr& intr)
             * @brief
             * @param[in] rgb_image, the new frame to integrate
             * @param[in] depth_map, the raycasted depth map
             * @param[in] camera_pose, the current pose of the camera
             * @param[in] intr, the intrinsic parameters of the RGB camera
             */
            virtual void integrate(const Image& rgb_image, const Dists& depth_map, const Affine3f& camera_pose, const Intr& intr);

            /**
             * @name swap
             * @fn void swap(CudaData& data)
             * @brief Swap memory content
             */
            void swap(CudaData& data) {data_.swap(data);}

            /**
             * @name fetchColors
             * @fn void fetchColors(const DeviceArray<Point>& cloud, DeviceArray<RGB>& colors) const
             * @brief Gets color for a point cloud
             * @param[in] cloud, the coordinates of the colors to extract
             * @param[in] colors, the colors stored in the volume
             */
            void fetchColors(const DeviceArray<Point>& cloud, DeviceArray<RGB>& colors) const;

        private:

#pragma mark -
#pragma mark Private attributes

            /**< Memory on the device */
            CudaData data_;
            /**< The truncation distance of the volume */
            float trunc_dist_;
            /**< Maximum weight */
            int max_weight_;
            /**< Number of voxels along each coordinates */
            Vec3i dims_;
            /**< Size of the volume along each coordinates */
            Vec3f size_;
            /**< The pose of the volume in the world */
            Affine3f pose_;
        };
    }
}

#endif //KFUSION_COLOR_VOLUME_HPP
