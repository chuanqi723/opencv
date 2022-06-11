/*
 *  @file: photo_decode.c
 *  @brief:
 *  @version: 0.0
 *  @author:
 *  @date: 2018/11/20
 */
/******************************************************************************
@note
    Copyright 2017, Megvii Corporation, Limited
                            ALL RIGHTS RESERVED
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "sdk_config.h"
#include "hisi_open.h"
#include "demo_comm.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"



/************************** C FUNC **************************/
#ifdef __cplusplus
extern "C" {
#endif  //@__cplusplus

#define ALIGN_DOWN(x, a)         ( ( (x) / (a)) * (a) )
void extent_face_rect(MGVL1_RECT_S *rect, float scale, int frame_width, int frame_height);

int decode_photo_to_nv21(char *jpg_file_path, COMMON_NV21_IMAGE_S *nv21_frame, int max_width, int max_height, float *zoom_ratio)
{
    int size = 0;
    float zoom = 1.0;
    
    if(!jpg_file_path || !nv21_frame)
        return -1;

    cv::Mat image_resized;
    cv::Mat image = cv::imread(jpg_file_path, cv::IMREAD_COLOR);
    if(image.empty())
    {
        printf("image is empty\n");
        return -1; //是否加载成功
    }

    if(max_width <= 0)
        max_width = image.cols;
    if(max_height <= 0)
        max_height = image.rows;

    max_width = ALIGN_DOWN(max_width, 2);
    max_height = ALIGN_DOWN(max_height, 2);

    if(image.rows <= max_height && image.cols <= max_width)
    {
        image_resized = image;
        zoom = 1.0;
    }
    else
    {
        zoom = 1.0 * max_width / image.cols;
        if(zoom > 1.0 * max_height / image.rows)
        {
            zoom = 1.0 * max_height / image.rows;
        }
        float new_w = image.cols * (zoom);
        float new_h = image.rows * (zoom);
        cv::resize(image, image_resized, cv::Size(new_w, new_h));
        if(image_resized.empty())
            return -1;

        if(!image_resized.data)
            return -1;
    }

    cv::Rect rect(0, 0, ALIGN_DOWN((int)image_resized.cols, 2), ALIGN_DOWN((int)image_resized.rows, 2));
    cv::Mat image_roi = image_resized(rect);
    if(image_roi.empty())
    {
        printf("buf empty \n");
        return -1;
    }

    cv::Mat dst_image;
    cvtColor(image_roi, dst_image, cv::COLOR_BGR2YUV_I420);
    if(dst_image.empty())
    {
        printf("buf empty \n");
        return -1;
    }

    size = image_roi.rows * image_roi.cols;
    memset(nv21_frame, 0, sizeof(COMMON_NV21_IMAGE_S));
    nv21_frame->data = (unsigned char *)malloc(size * 3 / 2);

    if(nv21_frame->data == NULL)
    {
        return -1;
    }
    nv21_frame->width = image_roi.cols;
    nv21_frame->height = image_roi.rows;
    memcpy(nv21_frame->data, dst_image.ptr(), size);

    unsigned char *p = nv21_frame->data + size;
    unsigned char *u = dst_image.ptr() + size;
    unsigned char *v = dst_image.ptr() + size + size / 4;
    size = image_roi.rows * image_roi.cols / 4;

    for(int i = 0; i < size; i++)
    {
        p[i * 2] = v[i];
        p[i * 2 + 1] = u[i];
    }

    zoom = 1.0 * image_resized.cols / image.cols;
    if(zoom_ratio)
        *zoom_ratio = zoom;
    return 0;
}

int save_nv21_jpeg(char *jpg_file_path, char *nv21_addr, int nv21_width, int nv21_height)
{
    if(!jpg_file_path || !nv21_addr || nv21_width <= 0 || nv21_height <= 0)
        return -1;

    cv::Mat image_nv21(nv21_height + nv21_height / 2, nv21_width, CV_8UC1, (unsigned char *)nv21_addr);
    cv::Mat image_bgr;
    cvtColor(image_nv21, image_bgr, cv::COLOR_YUV2BGR_NV21);
    if(image_bgr.empty())
    {
        printf("image_bgr empty \n");
        return -1;
    }

    imwrite(jpg_file_path, image_bgr);
    return 0;
}

int save_rect_jpg_from_nv21(char *jpg_file_path, char *nv21_addr, int nv21_width, int nv21_height, MGVL1_RECT_S rect, float rect_extent_ratio)
{
    if(!jpg_file_path || !nv21_addr || nv21_width <= 0 || nv21_height <= 0 || rect_extent_ratio < 0.0)
        return -1;

    cv::Mat image_nv21(nv21_height + nv21_height / 2, nv21_width, CV_8UC1, (unsigned char *)nv21_addr);
    cv::Mat image_bgr;
    cvtColor(image_nv21, image_bgr, cv::COLOR_YUV2BGR_NV21);
    if(image_bgr.empty())
    {
        printf("image_bgr empty \n");
        return -1;
    }
    
    if(rect_extent_ratio != 0.0)
    {
        extent_face_rect(&rect, rect_extent_ratio, nv21_width, nv21_height);
    }
    
    cv::Mat image_rect = image_bgr(cv::Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top));
    if(image_rect.empty())
        return -1;

    imwrite(jpg_file_path, image_rect);
    return 0;
}

int crop_rect_from_nv21(COMMON_NV21_IMAGE_S *src_image, COMMON_NV21_IMAGE_S *dst_image, MGVL1_RECT_S crop_rect, float extent_ratio)
{
	int ret = -1;
	int image_size = 0;
	int row = 0, i = 0;
	int width = 0, height = 0;
	unsigned char *src_y = NULL, *dst_y=NULL;
	unsigned char *src_uv = NULL, *dst_uv=NULL;
    MGVL1_RECT_S real_rect = {0};
    
	if(!src_image || !dst_image || extent_ratio<0.0)
	{
		printf("input param is NULL[%p|%p|%f] !!!\n", src_image, dst_image, extent_ratio);
		ret = -1;
		goto EXIT;
	}

    src_y = src_image->data;
	src_uv = src_image->data+src_image->width*src_image->height;
    real_rect = crop_rect;
    extent_face_rect(&real_rect, extent_ratio, src_image->width, src_image->height);
	width = real_rect.right-real_rect.left+1;
	height = real_rect.bottom-real_rect.top+1;

	image_size = width * height * 3 /2;
	dst_image->data = (unsigned char*)malloc(sizeof(char) * image_size);
	if(!dst_image->data)
	{
		printf("malloc is error: %s !!!\n", strerror(errno));
		ret = -1;
		goto EXIT;
	}
	
	dst_y = dst_image->data;
	dst_uv = dst_y+(width*height);
	for(row = real_rect.top, i = 0; row <= real_rect.bottom; row++)
	{			
		memcpy(dst_y+i*width, src_y+real_rect.left+row*src_image->width, width);
		i++;
	}
    
	for(row = real_rect.top/2, i = 0; row < (real_rect.bottom+1)/2; row++)
	{
		memcpy(dst_uv+i*width, src_uv+real_rect.left+row*src_image->width,width);
		i++;
	}

	dst_image->width = width;
	dst_image->height = height;

	ret = 0;
    
EXIT:	
	return ret;
}

int crop_rect_from_jpg(char *jpg_file_path, char *rect_jpg_path, MGVL1_RECT_S rect, float extent_ratio)
{
    int center_x = 0, center_y = 0;
    int offset_x = 0, offset_y = 0;
    if(!jpg_file_path || !rect_jpg_path || extent_ratio < 0.0)
        return -1;

    cv::Mat image = cv::imread(jpg_file_path, cv::IMREAD_COLOR);
    if(image.empty())
        return -1;

    if(!image.data)
        return -1;

    if(rect.left < 0)
        rect.left = 0;
    else if(rect.left >= image.cols)
        return -1;

    if(rect.top < 0)
        rect.left = 0;
    else if(rect.top >= image.rows)
        return -1;

    if(rect.right <= 0)
        return -1;
    else if(rect.right >= image.cols)
        rect.right = image.cols - 1;

    if(rect.bottom <= 0)
        return -1;
    else if(rect.bottom >= image.rows)
        rect.bottom = image.rows - 1;

    center_x = (rect.left + rect.right) / 2;
    center_y = (rect.top + rect.bottom) / 2;
    offset_x = (rect.right - rect.left) * (1 + extent_ratio) / 2;
    offset_y = (rect.bottom - rect.top) * (1 + extent_ratio) / 2;
    rect.left = center_x - offset_x;
    if(rect.left < 0)
        rect.left = 0;

    rect.right = center_x + offset_x;
    if(rect.right >= image.cols)
        rect.right = image.cols - 1;

    rect.top = center_y - offset_y;
    if(rect.top < 0)
        rect.top = 0;

    rect.bottom = center_y + offset_y;
    if(rect.bottom >= image.rows)
        rect.bottom = image.rows - 1;

    cv::Mat rect_mat = image(cv::Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top));
    if(rect_mat.empty())
        return -1;

    if(!rect_mat.data)
        return -1;

    cv::imwrite(rect_jpg_path, rect_mat);
    return 0;
}

// 按照比例扩展人脸框
// scale [0.0, ~] 0.0表示不扩展
void extent_face_rect(MGVL1_RECT_S *rect, float scale, int frame_width, int frame_height)
{
    int face_w = 0, face_h = 0;
    int center_x = 0, center_y = 0;

    if(!rect || scale < 0)
    {
        printf("input param err !!!\n");
        return;
    }

    center_x = (rect->right + rect->left) / 2;
    center_y = (rect->top + rect->bottom) / 2;

    center_x &= ~3;     //@ 4的倍数
    center_y &= ~3;

    face_w = rect->right - rect->left;
    face_h = rect->bottom - rect->top;

    face_w *= 1 + scale;
    face_h *= 1 + scale;

    if(face_w & 0x1f)   //@ 取32倍数
        face_w = (face_w & ~0x1f) + 32;
    if(face_h & 0x1f)   //@ 取32倍数
        face_h = (face_h & ~0x1f) + 32;

    if(face_w < 64)
        face_w = 64;
    if(face_h < 64)
        face_h = 64;

    if(face_w > frame_width)
    {
        face_w = (frame_width & ~0x1f);
    }

    if(face_h > frame_height)
    {
        face_h = (frame_height & ~0x1f);
    }

    if(center_x + face_w / 2 > frame_width)
    {
        rect->right = frame_width;
        rect->left = rect->right - face_w;
    }
    else if(center_x - face_w / 2 < 0)
    {
        rect->left = 0;
        rect->right = rect->left + face_w;
    }
    else
    {
        rect->left = center_x - face_w / 2;
        rect->right = rect->left + face_w;
    }

    if(center_y + face_h / 2 > frame_height)
    {
        rect->bottom = frame_height;
        rect->top = rect->bottom - face_h;
    }
    else if(center_y - face_h / 2 < 0)
    {
        rect->top = 0;
        rect->bottom = rect->top + face_h;
    }
    else
    {
        rect->top = center_y - face_h / 2;
        rect->bottom = rect->top + face_h;
    }
}
#include <vector>
using namespace std;
int save_nv21_png(char *png_file_path, char *nv21_addr, int nv21_width, int nv21_height)
{
    if(!png_file_path || !nv21_addr || nv21_width <= 0 || nv21_height <= 0)
        return -1;

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    cv::Mat image_nv21(nv21_height + nv21_height / 2, nv21_width, CV_8UC1, (unsigned char *)nv21_addr);
    cv::Mat image_bgr;
    cvtColor(image_nv21, image_bgr, CV_YUV2BGR_NV21);
    if(image_bgr.empty())
    {
        printf("image_bgr empty \n");
        return -1;
    }

    imwrite(png_file_path, image_bgr, compression_params);
    return 0;
}

int save_rect_nv21_png(char *png_file_path, char *nv21_addr, int nv21_width, int nv21_height, MGVL1_RECT_S rect, float zoom_ratio)
{
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

  if(!png_file_path || !nv21_addr || nv21_width <= 0 || nv21_height <= 0 || zoom_ratio < 0.0)
        return -1;

    cv::Mat image_nv21(nv21_height + nv21_height / 2, nv21_width, CV_8UC1, (unsigned char *)nv21_addr);
    cv::Mat image_bgr;
    cvtColor(image_nv21, image_bgr, cv::COLOR_YUV2BGR_NV21);
    if(image_bgr.empty())
    {
        printf("image_bgr empty \n");
        return -1;
    }

    if(zoom_ratio > 0.001)
    {
      extent_face_rect(&rect, zoom_ratio, nv21_width, nv21_height);
    }

    cv::Mat image_rect = image_bgr(cv::Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top));
    if(image_rect.empty())
        return -1;

    imwrite(png_file_path, image_rect, compression_params);
    return 0;
}



#ifdef __cplusplus
}
#endif //@__cplusplus
