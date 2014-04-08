#pragma warning (disable :4996)
#undef _CRT_SECURE_NO_DEPRECATE
#include <OpenNI.h>
#include <PS1080.h> // For XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE property
#include <io/capture.hpp>

using namespace std;
using namespace openni;

#define REPORT_ERROR(msg) kfusion::cuda::error ((msg), __FILE__, __LINE__)

double
calculateFocalLength (int pixelWidth, double fov)
{
  return pixelWidth / (2.0 * tan (fov / 2.0));
}

struct kfusion::OpenNISource::Impl
{
  openni::Device device;
  openni::VideoStream depthStream;
  openni::VideoStream colorStream;
  openni::VideoFrameRef depthFrame;
  openni::VideoFrameRef colorFrame;
  char strError[1024];
  bool has_depth;
  bool has_image;
};

kfusion::OpenNISource::OpenNISource() 
  : depth_focal_length_VGA (0.f)
  , baseline (0.f)
  , shadow_value (0)
  , no_sample_value (0)
  , pixelSize (0.0)
  , max_depth (0)
{}

kfusion::OpenNISource::OpenNISource(int device) {open (device); }
kfusion::OpenNISource::OpenNISource(const string& filename) {open (filename); }
kfusion::OpenNISource::~OpenNISource() { release (); }

void kfusion::OpenNISource::open (int device)
{
  open(""); // An empty string triggers openni::ANY_DEVICE
}

// uri can be a device ID or filename
void kfusion::OpenNISource::open(const std::string& filename)
{
    impl_ = cv::Ptr<Impl>( new Impl () );

    openni::Status rc;

    rc = openni::OpenNI::initialize ();
    if (rc != openni::STATUS_OK)
    {
      const string error(openni::OpenNI::getExtendedError());
      sprintf (impl_->strError, "Init failed: %s\n", error.c_str() );
      REPORT_ERROR (impl_->strError);
    }

    if (filename.length() > 0)
      rc = impl_->device.open( filename.c_str() );
    else
      rc = impl_->device.open(openni::ANY_DEVICE);

    if (rc != openni::STATUS_OK)
    {
      const string error(openni::OpenNI::getExtendedError());
      sprintf (impl_->strError, "Device open failed: %s\n", error.c_str() );
      REPORT_ERROR (impl_->strError);
    }

    impl_->has_image = impl_->device.hasSensor(SENSOR_COLOR);
    impl_->has_depth = impl_->device.hasSensor(SENSOR_DEPTH);

    rc = impl_->device.setDepthColorSyncEnabled(true);
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }

    openni::VideoMode colorMode;
    colorMode.setResolution (640, 480);
    colorMode.setFps(30);
    colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

    openni::VideoMode depthMode(colorMode);
    depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

    rc = impl_->colorStream.create(impl_->device, openni::SENSOR_COLOR);
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }

    rc = impl_->colorStream.setVideoMode(colorMode);
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }

    rc = impl_->depthStream.create(impl_->device, openni::SENSOR_DEPTH);
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }

    rc = impl_->depthStream.setVideoMode(depthMode);
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }

    getParams ();


    rc = impl_->colorStream.start();
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }

    rc = impl_->depthStream.start();
    if (rc != openni::STATUS_OK)
    {
      sprintf (impl_->strError, "Init failed: %s\n", openni::OpenNI::getExtendedError() );
      REPORT_ERROR (impl_->strError);
    }
}

void kfusion::OpenNISource::release ()
{
    if (impl_)
    {
      impl_->colorStream.stop();
      impl_->colorStream.destroy();

      impl_->depthStream.stop();
      impl_->depthStream.destroy();

      impl_->device.close();
    }

    impl_.release();
    depth_focal_length_VGA = 0;
    baseline = 0.f;
    shadow_value = 0;
    no_sample_value = 0;
    pixelSize = 0.0;
}

bool kfusion::OpenNISource::grab(cv::Mat& depth, cv::Mat& color)
{
    Status rc = STATUS_OK;

    if (impl_->has_depth)
    {
      rc = impl_->depthStream.readFrame(&impl_->depthFrame);
      if (rc != openni::STATUS_OK)
      {
        sprintf (impl_->strError, "Frame grab failed: %s\n", openni::OpenNI::getExtendedError() );
        REPORT_ERROR (impl_->strError);
      }

      const void* pDepth = impl_->depthFrame.getData();
      int x = impl_->depthFrame.getWidth();
      int y = impl_->depthFrame.getHeight();
      cv::Mat(y, x, CV_16U, (void*)pDepth).copyTo(depth);
    }
    else
    {
        depth.release();
        printf ("no depth\n");
    }

    if (impl_->has_image)
    {
      rc = impl_->colorStream.readFrame(&impl_->colorFrame);
      if (rc != openni::STATUS_OK)
      {
        sprintf (impl_->strError, "Frame grab failed: %s\n", openni::OpenNI::getExtendedError() );
        REPORT_ERROR (impl_->strError);
      }

      const void* pColor = impl_->colorFrame.getData();
      int x = impl_->colorFrame.getWidth();
      int y = impl_->colorFrame.getHeight();
      cv::Mat(y, x, CV_8UC3, (void*)pColor).copyTo(color);
    }
    else
    {
        color.release();
        printf ("no color\n");
    }

    return impl_->has_image || impl_->has_depth;
}

void kfusion::OpenNISource::getParams ()
{
    openni::Status rc = STATUS_OK;

    max_depth = impl_->depthStream.getMaxPixelValue();

    double baseline_local = 0;    // in mm
    if ( impl_->depthStream.isPropertySupported (XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE) )
    {
      rc = impl_->depthStream.getProperty (XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE, &baseline_local); // Device specific -- from PS1080.h
    }
    else
      printf ("Baseline not available.\n");

    shadow_value = 0;
    no_sample_value = 0;

    // baseline from cm -> mm
    baseline = (float)(baseline_local * 10);

    //focal length
    int width = impl_->colorStream.getVideoMode().getResolutionX();
    float hFov = impl_->colorStream.getHorizontalFieldOfView();
    depth_focal_length_VGA = (float) calculateFocalLength( width, hFov);
}

bool kfusion::OpenNISource::setRegistration (bool value)
{
    Status rc = STATUS_OK;

    if(value) // "on"
    {
      rc = impl_->device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
      if (rc != openni::STATUS_OK)
      {
        sprintf (impl_->strError, "Setting registration failed: %s\n", openni::OpenNI::getExtendedError() );
        REPORT_ERROR (impl_->strError);
      }
    }
    else // "off"
    {
      rc = impl_->device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_OFF);
      if (rc != openni::STATUS_OK)
      {
        sprintf (impl_->strError, "Setting registration failed: %s\n", openni::OpenNI::getExtendedError() );
        REPORT_ERROR (impl_->strError);
      }
    }

    getParams ();
    return rc == STATUS_OK;
}
