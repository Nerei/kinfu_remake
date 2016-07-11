/**
 *  @file   bin_grabber.hpp
 *  @brief  Bin Grabber for PCL KinFu app
 *
 *  @author Gabriel Cuendet
 *  @date   21/06/2016
 *  Copyright (c) 2016 Gabriel Cuendet. All rights reserved.
 */

#ifndef __KFUSION_BIN_SOURCE__
#define __KFUSION_BIN_SOURCE__

#include <kfusion/kinfu.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <fstream>

namespace kfusion {
class KF_EXPORTS BinSource {
 public:
  typedef kfusion::PixelRGB RGB24;
  typedef kfusion::RGB RGB32;

  struct KinectData {
    /**< Depth map width */
    int depth_frame_width;
    /**< Depth map height */
    int depth_frame_height;
    /**< Size in bytes of one frame in the binary file */
    int depth_block_size;
    /**< Array of memory for the depth map */
    std::vector<unsigned short> depth_frame;
    /**< RGB Image width */
    int rgb_frame_width;
    /**< RGB Image height */
    int rgb_frame_height;
    /**< Size in bytes of one frame in the binary file */
    int rgb_block_size;
    /**< Array of memory for the rgb image */
    std::vector<RGB32> rgb_frame;

    /**
     * @name  KinectData
     * @fn  KinectData(void)
     * @brief Default constructor
     */
    KinectData(void) : depth_frame_width(0), depth_frame_height(0),
                       depth_frame(0), rgb_frame_width(0), rgb_frame_height(0), rgb_frame(0) {}

    /**
     * @name  ~KinectData
     * @fn  ~KinectData(void)
     * @brief Default destructor
     */
    ~KinectData(void) {}

    /**
     * @name  IsDepthFrameOk
     * @fn  bool IsDepthFrameOk(void) const
     * @brief Check that the depth frame is NOT empty and that its size
     *        corresponds to its expected width and height
     * @return true if not empty and correct dimensions, false otherwise
     */
    bool IsDepthFrameOk(void) const {
      return (depth_frame.size() != 0) &&
             (depth_frame.size() == depth_frame_width * depth_frame_height);
    }

    /**
     * @name  IsRgbFrameOk
     * @fn  bool IsRgbFrameOk(void) const
     * @brief Check that the RGB frame is NOT empty and that its size
     *        corresponds to its expected width and height
     * @return true if not empty and correct dimensions, false otherwise
     */
    bool IsRgbFrameOk(void) const {
      return (rgb_frame.size() != 0) &&
             (rgb_frame.size() == rgb_frame_width * rgb_frame_height);
    }
  };

  BinSource(const std::string& depth_filename, const std::string& rgb_filename,
            bool repeat = false);

  void open(const std::string& depth_filename, const std::string& rgb_filename,
            bool repeat = false);

  void release();

  ~BinSource();

  bool grab(cv::Mat &depth, cv::Mat &image);

  //parameters taken from camera/oni

  float depth_focal_length_VGA;
  float baseline;               // mm
  double pixelSize;             // mm
  unsigned short max_depth;     // mm

  bool setRegistration (bool value = false);

 private:

  /**< Data struct */
  cv::Ptr<KinectData> kinect_data_;
  /**< Filestream of depth maps */
  std::ifstream depth_image_stream_;
  /**< Filestram of rgb images */
  std::ifstream rgb_image_stream_;
  /**< Current frame's index */
  int cur_frame_;
  /**< Total number of frames */
  int total_frames_;
  /**< Wether or not to manually align the depth maps and the color images */
  bool manual_align_;
  /**< Intrinsics and extrinsics parameters of the Kinect the bin files were recorded with */
  cv::Mat K_ir_;
  cv::Mat K_irInv_;
  cv::Mat K_rgb_;
  cv::Mat C_ir_;

  void getParams ();

};
}

#endif /* __KFUSION_BIN_SOURC__ */
