// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "kinfu_frame.hpp"

namespace cv {
namespace kinfu {


class TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolume(Point3i _res, float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
               float _raycastStepFactor, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, const Semantic& _semantic, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) = 0;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize,
                         cv::OutputArray points, cv::OutputArray normals, cv::OutputArray _voxelClass) const = 0;

    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0;
    virtual void reset() = 0;
        
    virtual ~TSDFVolume() { }
    
    //cv::Mat maxIndexMat;
    float voxelSize;
    float voxelSizeInv;
    Point3i volResolution;
    int maxWeight;
    cv::Affine3f pose;
    float raycastStepFactor;
    Point3f volSize;
    float truncDist;
    Vec4i volDims;
    Vec8i neighbourCoords;


    
    //-------------------------------------modification 
    bool updated_friction; 
    virtual void update_friction_tsdf(bool update, std::pair<int,int> x_y,int class_index, Eigen::Matrix<double, 7, 2> *dataSet, std::vector<double> measurements) = 0;
    virtual bool get_update_friction() = 0;
    virtual std::vector<int> get_x_y_class() = 0;
    int x_value;
    int y_value;
    int class_index;
    Eigen::Matrix<double, 7, 2> *dataSet;
    std::vector<double> measurements;
    //-------------------------------------modification 
};


cv::Ptr<TSDFVolume> makeTSDFVolume(Point3i _res, float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor);

} // namespace kinfu
} // namespace cv
#endif
