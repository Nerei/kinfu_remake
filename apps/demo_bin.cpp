#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
//#include <io/capture.hpp>
#include <io/bin_grabber.hpp>

using namespace kfusion;

struct KinFuApp
{
  /**
   * @name KeyboardCallback
   * @fn static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
   * @brief Define keyboard callback for the Kinfu app
   * @param[in] event
   * @param[in] pthis, void pointer on itself
   */
  static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
  {
      KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

      if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
          return;

      if(event.code == 't' || event.code == 'T')
          kinfu.take_cloud(*kinfu.kinfu_);

      if(event.code == 'i' || event.code == 'I')
          kinfu.interactive_mode_ = !kinfu.interactive_mode_;
  }

  /**
   * @name KinFuApp
   * @fn KinFuApp(BinSource& source)
   * @brief Constructor of the struct, only taking a BinSource data source
   * @param[in] source, a BinSource from which the frames are grabbed
   */
  KinFuApp(BinSource& source) : exit_ (false),  interactive_mode_(false), capture_ (source), pause_(false) {
      KinFuParams params = KinFuParams::default_params();
      kinfu_ = KinFu::Ptr( new KinFu(params) );

      capture_.setRegistration(true);

      cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
      viz.showWidget("cube", cube, params.volume_pose);
      viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
      viz.registerKeyboardCallback(KeyboardCallback, this);
  }

  /**
   * @name KinFuApp
   * @fn KinFuApp(BinSource& source, const KinFuParams& params)
   * @brief Constructor of the struct, allowing to initialize KinFu to different parameters
   * @param[in] source, a BinSource from which the frames are grabbed
   * @param[in] params, a KinFuParams struct containing the parameters
   */
  KinFuApp(BinSource& source, const KinFuParams& params) : exit_ (false),  interactive_mode_(false), capture_ (source), pause_(false) {
    kinfu_ = KinFu::Ptr( new KinFu(params) );

    capture_.setRegistration(true);

    cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
    viz.showWidget("cube", cube, params.volume_pose);
    viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
    viz.registerKeyboardCallback(KeyboardCallback, this);
  }

  /**
   * @name show_depth
   * @fn void show_depth(const cv::Mat& depth)
   * @brief Display the depth stream, after normalization
   * @param[in] depth, the depth image to normalize and display
   */
  void show_depth(const cv::Mat& depth) {
      cv::Mat display;
      //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
      depth.convertTo(display, CV_8U, 255.0/4000);
      cv::imshow("Depth", display);
  }

  /**
   * @name show_raycasted
   * @fn void show_raycasted(KinFu& kinfu)
   * @brief Show the reconstructed scene (using raycasting)
   * @param[in] kinfu instance
   */
  void show_raycasted(KinFu& kinfu) {
      const int mode = 3;
      if (interactive_mode_)
          kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
      else
          kinfu.renderImage(view_device_, mode);

      view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
      view_device_.download(view_host_.ptr<void>(), view_host_.step);
      cv::imshow("Scene", view_host_);
  }

  /**
   * @name take_cloud
   * @fn void take_cloud(KinFu& kinfu)
   * @brief Fetch cloud and display it
   * @param[in] kinfu instance
   */
  void take_cloud(KinFu& kinfu)
  {
      cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);

      cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);

      if (kinfu.params().integrate_color) {
          kinfu.color_volume()->fetchColors(cloud, color_buffer);
          cv::Mat color_host(1, (int)cloud.size(), CV_8UC4);
          cloud.download(cloud_host.ptr<Point>());
          color_buffer.download(color_host.ptr<RGB>());
          viz.showWidget("cloud", cv::viz::WCloud(cloud_host, color_host));
      } else
      {
          cloud.download(cloud_host.ptr<Point>());
          viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
          //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
      }
  }

  /**
   * @name execute
   * @fn bool execute()
   * @brief Run the main loop of the app
   * @return true if no error, false otherwise
   */
  bool execute()
  {
      KinFu& kinfu = *kinfu_;
      cv::Mat depth, image;
      double time_ms = 0;
      bool has_image = false;

      for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
      {
          bool has_frame = capture_.grab(depth, image);
          if (!has_frame)
              return std::cout << "Can't grab" << std::endl, false;

          depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
          color_device_.upload(image.data, image.step, image.rows, image.cols);

          {
              SampledScopeTime fps(time_ms); (void)fps;
              if (kinfu.params().integrate_color)
                  has_image = kinfu(depth_device_, color_device_);
              else
                  has_image = kinfu(depth_device_);
          }

          if (has_image)
              show_raycasted(kinfu);

          show_depth(depth);
          if (kinfu.params().integrate_color)
              cv::imshow("Image", image);

          if (!interactive_mode_)
              viz.setViewerPose(kinfu.getCameraPose());

          int key = cv::waitKey(pause_ ? 0 : 3);

          switch(key)
          {
          case 't': case 'T' : take_cloud(kinfu); break;
          case 'i': case 'I' : interactive_mode_ = !interactive_mode_; break;
          case 27: exit_ = true; break;
          case 32: pause_ = !pause_; break;
          }

          //exit_ = exit_ || i > 100;
          viz.spinOnce(3, true);
      }
      return true;
  }

  /**< */
  bool pause_; // = false
  /**< Stop the execution when set to true */
  bool exit_;
  /**< Allow for free point of view (otherwise, follows the camera) */
  bool interactive_mode_;
  /**< Reference to a source of data BinSource */
  BinSource& capture_;
  /**< Pointer to the instance of kinfu */
  KinFu::Ptr kinfu_;
  /**< */
  cv::viz::Viz3d viz;

  /**< View of the scene (raycasting) */
  cv::Mat view_host_;
  /**< View of the scene on the GPU */
  cuda::Image view_device_;
  /**< Depth frame on the GPU */
  cuda::Depth depth_device_;
  /**< Color frame on the GPU */
  cuda::Image color_device_;
  /**< */
  cuda::DeviceArray<Point> cloud_buffer;
  /**< */
  cuda::DeviceArray<RGB> color_buffer;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
  int device = 0;
  cuda::setDevice (device);
  cuda::printShortCudaDeviceInfo (device);

  if(cuda::checkIfPreFermiGPU(device))
      return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

  BinSource capture(argv[1], argv[2]);

  KinFuParams custom_params = KinFuParams::default_params();
  custom_params.integrate_color = true;
  custom_params.volume_dims = Vec3i::all(256);
  custom_params.volume_size = Vec3f::all(0.7f);
  custom_params.volume_pose = Affine3f().translate(Vec3f(-custom_params.volume_size[0]/2, -custom_params.volume_size[1]/2, 0.5f));
  custom_params.intr = Intr(520.89, 520.23, 324.54, 237.553);
  custom_params.tsdf_trunc_dist = 0.05;

  KinFuApp app (capture, custom_params);

  // executing
  try { app.execute (); }
  catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
  catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

  return 0;
}
