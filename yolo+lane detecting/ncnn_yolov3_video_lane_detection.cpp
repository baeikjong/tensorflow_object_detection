// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN
void splitTwoSideLines(std::vector<cv::Vec4i> &lines, std::vector<std::vector<float>> &lefts, std::vector<std::vector<float>> &rights, float slope_threshold)
{
    int i;
    lefts.clear();
    rights.clear();
    std::vector<float> temp;
    for( i = 0 ; i < lines.size() ; i++ )
    {
        temp.clear();
        cv::Vec4i line = lines[i];
        int x1, y1, x2, y2;
        x1 = line[0];
        y1 = line[1];
        x2 = line[2];
        y2 = line[3];
        if (x1 - x2 == 0)
            continue;
        float slope = (float)(y2-y1)/(float)(x2-x1);
        if (abs(slope) < slope_threshold)
            continue;
        if( slope <= 0)
        {
            temp.push_back(slope);
            temp.push_back(x1);
            temp.push_back(y1);
            temp.push_back(x2);
            temp.push_back(y2);
            lefts.push_back(temp);
        }
        else
        {
            temp.push_back(slope);
            temp.push_back(x1);
            temp.push_back(y1);
            temp.push_back(x2);
            temp.push_back(y2);
            rights.push_back(temp);
        }
    }
    return;
}
bool comp(std::vector<float> a, std::vector<float> b)
{
    return (a[0] > b[0]);
}
void medianPoint(std::vector<std::vector<float>> &lines, std::vector<float> &line)
{
    line.clear();
    size_t size = lines.size();
    if (size == 0)
        return;
    sort(lines.begin(), lines.end(), comp);
    line = lines[(int)(size/2.0)];
    return;
}
int interpolate(int x1, int y1, int x2, int y2, int y)
{
    return int(float(y - y1) * float(x2-x1) / float(y2-y1) + x1);
}

void lineFitting(cv::Mat &image, cv::Mat &result, std::vector<cv::Vec4i> &lines, cv::Scalar color, int thickness, float slope_threshold)
{
    image.copyTo(result);
//    result = imageCopy(image);
    int height = image.rows;
    std::vector<std::vector<float>> lefts, rights;
    splitTwoSideLines(lines, lefts, rights, slope_threshold);
    std::vector<float> left, right;
    medianPoint(lefts, left);
    medianPoint(rights, right);
    int min_y = int(height * 0.6);
    int max_y = height;
    if( !left.empty()) 
    {
        int min_x_left = interpolate(left[1], left[2], left[3], left[4], min_y);
        int max_x_left = interpolate(left[1], left[2], left[3], left[4], max_y);
        cv::line(result, cv::Point(min_x_left, min_y), cv::Point(max_x_left, max_y), color, thickness);
    }
    if( !right.empty())
    {
        int min_x_right = interpolate(right[1], right[2], right[3], right[4], min_y);
        int max_x_right = interpolate(right[1], right[2], right[3], right[4], max_y);
        cv::line(result, cv::Point(min_x_right, min_y), cv::Point(max_x_right, max_y), color, thickness);
    }
    return;
}
cv::Mat makeBlackImage(cv::Mat &image, bool color)
{
    if(color)
        return cv::Mat::zeros(image.size(), CV_8UC3);
    else
        return cv::Mat::zeros(image.size(), image.type());
}
cv::Mat fillPolyROI(cv::Mat &image, std::vector<cv::Point> points)
{
    cv::Mat result = makeBlackImage(image, false);
    std::vector<std::vector<cv::Point> > fillContAll;
    fillContAll.push_back(points);
    if(image.channels()==1)
        cv::fillPoly(result, fillContAll, cv::Scalar(255));
    else 
        cv::fillPoly(result, fillContAll, cv::Scalar(255, 255, 255));
    return result;
}
void polyROI(cv::Mat &image, cv::Mat &result, std::vector<cv::Point> points) 
{
    result = fillPolyROI(image, points);
    cv::bitwise_and(result, image, result);
    return;
}
void imageHoughLinesP(cv::Mat &image, std::vector<cv::Vec4i> &lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap) 
{
    lines.clear();
    cv::HoughLinesP(image,lines,rho,theta,threshold, minLineLength, maxLineGap);
    return;
}
void lane_detection(cv::Mat &image, cv::Mat &result)
{
    image.copyTo(result);
    cv::Mat result_gray, result_edge, result_roi, image_lane;
    std::vector<cv::Vec4i> lines;
    cv::cvtColor(result, result_gray, cv::COLOR_BGR2GRAY);
    cv::Canny(result_gray,result_edge, 100, 200);
    int height = result.rows;
    int width = result.cols;
    cv::Point pt1, pt2, pt3, pt4;
    std::vector<cv::Point> roi_corners;
    pt1 = cv::Point(width*0.45, height*0.65);
    pt2 = cv::Point(width*0.55, height*0.65);
    pt3 = cv::Point(width, height*1.0);
    pt4 = cv::Point(0, height*1.0);
    roi_corners.push_back(pt1);
    roi_corners.push_back(pt2);
    roi_corners.push_back(pt3);
    roi_corners.push_back(pt4);
    polyROI(result_edge,result_roi, roi_corners);
    imageHoughLinesP(result_roi, lines, 1.0, 3.1415/180., 30, 20,100);
    lineFitting(image, result, lines, cv::Scalar(0, 0, 255), 5, 10.*3.1415/180.);
    return;
}
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov3;

#if NCNN_VULKAN
    yolov3.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    yolov3.load_param("mobilenetv2_yolov3.param");
    yolov3.load_model("mobilenetv2_yolov3.bin");

    const int target_size = 352;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

//     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();
    lane_detection(image, image);
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
	image.copyTo(bgr);
    //cv::imshow("image", image);
    //cv::waitKey(0);
}

void imageProcessing(cv::Mat &m, cv::Mat &result)
{
	cv::Mat temp;
	m.copyTo(temp);
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN
	
	std::vector<Object> objects;
    detect_yolov3(temp, objects);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    draw_objects(temp, objects);
    temp.copyTo(result);
    return;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    const char* videopath = argv[1];
	
	cv::VideoCapture cap(videopath);
    if(cap.isOpened())
    {
        printf("Video Opened\n");
    }
    else
    {
        printf("Video Not Opened\n");
        printf("Program Abort\n");
        exit(-1);
    }
    std::string savePath = "output.mp4";
    int fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fourcc = cap.get(cv::CAP_PROP_FOURCC);
    cv::VideoWriter out(savePath.c_str(), fourcc, fps, cv::Size(width, height), true);
    cv::namedWindow("Input", cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("Output", cv::WINDOW_GUI_EXPANDED);
    cv::Mat frame, output;
    while(cap.read(frame))
    {
        imageProcessing(frame, output);
        out.write(output);
        cv::imshow("Input", frame);
        cv::imshow("Output", output);
        char c = (char)cv::waitKey(int(1000.0/fps));
        if (c==27)
            break;
    }
    cap.release();
    out.release();
    cv::destroyAllWindows();
    
    return 0;
}