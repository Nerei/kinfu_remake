/**
 *  @file   bin_grabber.hpp
 *  @brief  Bin Grabber for PCL KinFu app
 *
 *  @author Gabriel Cuendet
 *  @date   21/06/2016
 *  Copyright (c) 2016 Gabriel Cuendet. All rights reserved.
 */

#include <iostream>
#include <kfusion/kinfu.hpp>
#include <fstream>

#include "io/bin_grabber.hpp"

namespace kfusion {

  BinSource::BinSource(const std::string& depth_filename, const std::string& rgb_filename,
            bool repeat)
    : manual_align_(false)
    , K_ir_((cv::Mat_<float>(3,3) << 589.9,   0.0, 328.4,
                                      0.0, 589.1, 236.89,
                                      0.0,   0.0,   1.0))
    , K_rgb_((cv::Mat_<float>(3,3) << 520.9,   0.0, 324.54,
                                       0.0, 520.2, 237.55,
                                       0.0,   0.0,   1.0))
    , C_ir_((cv::Mat_<float>(3,1) << -25.4, -0.13, -2.18)) // Ref: http://wiki.ros.org/kinect_calibration/technical
  {
    cv::invert(K_ir_, K_irInv_); // R_ir is Identity

    kinect_data_ = cv::Ptr<BinSource::KinectData>(new BinSource::KinectData());
    this->open(depth_filename, rgb_filename, repeat);
  }

  void BinSource::open(const std::string& depth_filename, const std::string& rgb_filename,
                       bool repeat) {

    depth_image_stream_.open(depth_filename.c_str(), std::ios::in | std::ios::binary);
    if(!depth_image_stream_.is_open()) {
      std::cerr << "[kfusion::BinSource::open] Error: Path is not a regular file: " << depth_filename << std::endl;
    }
    rgb_image_stream_.open(rgb_filename.c_str(), std::ios::in | std::ios::binary);
    if (!rgb_image_stream_.is_open()) {
      std::cerr << "[kfusion::BinSource::open] Error: Path is not a regular file: " << rgb_filename << std::endl;
    }

    int rgb_num_frames, rgb_block_size;
    int rgb_frame_w, rgb_frame_h;

    rgb_image_stream_.read((char*)&rgb_num_frames, sizeof(int));
    rgb_image_stream_.read((char*)&rgb_block_size, sizeof(int)); // size in bytes for 1 frame
    rgb_image_stream_.read((char*)&rgb_frame_w, sizeof(int));
    rgb_image_stream_.read((char*)&rgb_frame_h, sizeof(int));

    int depth_num_frames, depth_block_size;
    int depth_frame_w, depth_frame_h;

    depth_image_stream_.read((char*)&depth_num_frames, sizeof(int));
    depth_image_stream_.read((char*)&depth_block_size, sizeof(int)); // size in bytes for 1 frame
    depth_image_stream_.read((char*)&depth_frame_w, sizeof(int));
    depth_image_stream_.read((char*)&depth_frame_h, sizeof(int));

    if (depth_num_frames != rgb_num_frames){
      std::cerr << "[kfusion::BinSource::open] : Watch out not same amount of depth and rgb frames: #depth frames = " <<
      depth_num_frames << " ; #rgb frames = " << rgb_num_frames << std::endl;
    }

    if (depth_num_frames > 0){
      // TODO: Better way to handle different number of frames...
      total_frames_ = std::min(depth_num_frames, rgb_num_frames);

      kinect_data_->depth_block_size = depth_block_size;
      kinect_data_->depth_frame_width = depth_frame_w;
      kinect_data_->depth_frame_height = depth_frame_h;
    } else {
      std::cerr << "[kfusion::BinSource::open] : no depth frames in the binary file" << std::endl;
    }

    if (rgb_num_frames > 0){
      kinect_data_->rgb_block_size = rgb_block_size;
      kinect_data_->rgb_frame_width = rgb_frame_w;
      kinect_data_->rgb_frame_height = rgb_frame_h;
    } else {
      std::cerr << "[kfusion::BinSource::open] : no rgb frames in the binary file" << std::endl;
    }
  }

  void BinSource::release() {
    kinect_data_.release();
  }

  BinSource::~BinSource() {
    this->release();
  }

  bool BinSource::grab(cv::Mat &depth, cv::Mat &image) {
    bool success = false;

    if (cur_frame_ < total_frames_) {
      unsigned short crap;

      int depth_frame_size = this->kinect_data_->depth_frame_width * this->kinect_data_->depth_frame_height;
      this->kinect_data_->depth_frame.resize(depth_frame_size);
      for (int i=0; i < depth_frame_size; ++i) {
        depth_image_stream_.read((char*)(&crap), 2);
        depth_image_stream_.read((char*)(&(this->kinect_data_->depth_frame[i])), 2);
      }

      int rgb_frame_size = this->kinect_data_->rgb_frame_width * this->kinect_data_->rgb_frame_height;
      this->kinect_data_->rgb_frame.resize(rgb_frame_size);
      rgb_image_stream_.read((char*)(&(this->kinect_data_->rgb_frame[0])), 4 * rgb_frame_size);

      image.create(this->kinect_data_->rgb_frame_height, this->kinect_data_->rgb_frame_width, CV_8UC4);
      memcpy(image.data, this->kinect_data_->rgb_frame.data(), 4 * rgb_frame_size);

      success = true;
      ++cur_frame_;

      if (manual_align_) {
        // Quick check on the data
        if (!this->kinect_data_->IsDepthFrameOk()) {
          std::cerr << "[BinSource::grab] ERROR: " <<
          "Depth frame is not loaded properly (size does not correspond or is empty)" << std::endl;
          return false;
        }

        int depth_size = this->kinect_data_->depth_frame.size();
        int depth_width = this->kinect_data_->depth_frame_width;

        // Allocate space for the new depth frame and initialize at 0
        std::vector<unsigned short> aligned_depth_frame(depth_size, 0);

        int d_i;
        #pragma omp parallel for
        for (d_i = 0; d_i < depth_size; ++d_i) {
          float u_ir = d_i % depth_width;
          float v_ir = d_i / depth_width;

          cv::Mat uvw_ir = (cv::Mat_<float>(3,1) << u_ir, v_ir, 1.0);
          // Backproject to world
          cv::Mat xyz = K_irInv_ * uvw_ir;

          // Reproject on RGB camera
          cv::Mat uvw_rgb = K_rgb_ * ((static_cast<float>(this->kinect_data_->depth_frame[d_i]) * xyz) - C_ir_);

          // Fill in the new depth frame
          int u_rgb = (int)round(uvw_rgb.at<float>(0) / uvw_rgb.at<float>(2));
          int v_rgb = (int)round(uvw_rgb.at<float>(1) / uvw_rgb.at<float>(2));
          if (u_rgb < 0 || u_rgb > depth_width - 1) {
            continue;
          }
          if (v_rgb < 0 || v_rgb > this->kinect_data_->depth_frame_height - 1) {
            continue;
          }
          int j = v_rgb * depth_width + u_rgb;
          unsigned short new_depth_val = static_cast<unsigned short>(uvw_rgb.at<float>(2));


          if (new_depth_val > 0) {
            #pragma omp critical
            {
              // Check that new depth is smaller than current depth (not masked)
              if (aligned_depth_frame[j] > 0) {
                if (new_depth_val < aligned_depth_frame[j])
                  aligned_depth_frame[j] = new_depth_val;
              } else {
                aligned_depth_frame[j] = new_depth_val;
              }
            }
          }
        }

        // Swap the content of depth_frame_ and aligned_depth_frame
        this->kinect_data_->depth_frame.swap(aligned_depth_frame);
      }
      depth.create(this->kinect_data_->depth_frame_height, this->kinect_data_->depth_frame_width, CV_16UC1);
      memcpy(depth.data, this->kinect_data_->depth_frame.data(), 2 * depth_frame_size);
    }

    return success;
  }

  //parameters taken from camera/oni

  bool BinSource::setRegistration (bool value) {
    manual_align_ = value;
  }

  void BinSource::getParams () {}
}
