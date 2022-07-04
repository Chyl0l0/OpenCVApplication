// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <deque>
void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void incBrightnessAdd()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		const int increaseValue = 50;
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int increment = val + increaseValue;
				dst.at<uchar>(i, j) = max(0,min(increment,255));
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void incBrightnessMul()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		const int increaseValue = 5;
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int increment = val * increaseValue;
				dst.at<uchar>(i, j) = max(0, min(increment, 255));
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void fourColors()
{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat img(256, 256, CV_8UC3);

		// Get the current time again and compute the time difference [s]
		
		for (int i = 0; i < 256; i++)
		{
			for (int j = 0; j < 256; j++)
			{
				if (i<128 && j<128)
				{
					img.at<Vec3b>(i, j)[0] = 255;
					img.at<Vec3b>(i, j)[1] = 255;
					img.at<Vec3b>(i, j)[2] = 255;
				}
				else if (i < 128 && j > 128)
				{
					img.at<Vec3b>(i, j)[0] = 0;
					img.at<Vec3b>(i, j)[1] = 0;
					img.at<Vec3b>(i, j)[2] = 255;
				}

				else if (i > 128 && j < 128)
				{
					img.at<Vec3b>(i, j)[0] = 0;
					img.at<Vec3b>(i, j)[1] = 255;
					img.at<Vec3b>(i, j)[2] = 0;
				}
				else
				{
					img.at<Vec3b>(i, j)[0] = 0;
					img.at<Vec3b>(i, j)[1] = 255;
					img.at<Vec3b>(i, j)[2] = 255;
				}

			}
		}
		t = ((double)getTickCount() - t) / getTickFrequency();

		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("output image", img);
		waitKey();
	
}

void inverseFloatMatrix()
{
	double t = (double)getTickCount(); // Get the current time [s]
	float vals[9] = { 1.2f, 2.3f, 3.f, 2.f, 3.f , 3.f , 4.f , 4.f , 4.f };
	Mat src(3, 3, CV_32FC1,vals);
	Mat dst;
	// Get the current time again and compute the time difference [s]
	t = ((double)getTickCount() - t) / getTickFrequency();
	dst = src.inv();
	// Print (in the console window) the processing time in [ms] 
	printf("Time = %.3f [ms]\n", t * 1000);
	cout << "Source:" << endl;
	cout << src << endl;
	cout << "Dest:" << endl;

	cout << dst << endl;
	imshow("src", src);
	waitKey();

}
void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void splitChannels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat channels[3];
	    split(src,channels);

		imshow("input image", src);
		imshow("R", channels[2]);
		imshow("G", channels[1]);
		imshow("B", channels[0]);
		waitKey();
	}
}

void convertGrayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat::zeros(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void convGrayscaleToBW()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat::zeros(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
			
				if (src.at<uchar>(i, j)>128)
				{
					dst.at<uchar>(i, j) = 255;

				}
				else
				{
					dst.at<uchar>(i, j) = 0;

				}
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void convBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);



		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				float b = (float)v3[0] / 255;
				float g = (float)v3[1] / 255;
				float r = (float)v3[2] / 255;

				float M = max(r, max(g, b));
				float m = min(r, min(g, b));
				float C = M - m;
				//value
				float V = M;
				//saturation
				float S = 0;
				if (V != 0)
					S = C / V;
				//hue
				float H = 0;
				if (C)
				{
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				if (H < 0)
				{
					H = H + 360;
				}
				/*
				if (H * 255 / 360 < 20.0f)
				{
					dstH.at<uchar>(i, j) = 0;
				}
				else
				{
					dstH.at<uchar>(i, j) = 255;

				}
				*/
				dstH.at<uchar>(i, j) = H * 255 / 360;
				dstS.at<uchar>(i, j) = S * 255 ;
				dstV.at<uchar>(i, j) = V * 255;
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

bool isInside(Mat img, int i, int j)
{
	int height = img.rows;
	int width = img.cols;
	if (i > 0 && i < height && j > 0 && j < width)
		return true;
	return false;
	

}

void testIsInside()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		int i, j;
	
		imshow("image", src);

		cout << "Height" << src.rows << " Width=" << src.cols <<endl;
		cout << "i=";
		cin >> i;
		cout << "j=";
		cin >> j;
		cout << isInside(src, i, j)<<endl;

		waitKey();

	}

}

void calculateHistogram(Mat src, int histogram[])
{
	memset(histogram, 0, 256 * sizeof(int));
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			histogram[src.at<uchar>(i, j)]++;
		}
	}
}
void testCalculateHistogram()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int histogram[256];
		calculateHistogram(src, histogram);
		for (int i = 0; i < 256; i++)
		{
			cout << histogram[i] << " ";
		}
		cout << endl;

		imshow("input image", src);
		waitKey();
	}
}

void calculateFDP(Mat src, float fdp[])
{
	memset(fdp, 0, 256 * sizeof(int));
	int height = src.rows;
	int width = src.cols;
	int size = height * width;
	int histogram[256] = { 0 };
	calculateHistogram(src, histogram);
	for (int i = 0; i < 256; i++)
	{
		fdp[i] =  histogram[i] / float(size);
	}

}
void testCalculateFDP()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		float fdp[256];
		calculateFDP(src, fdp);
		for (int i = 0; i < 256; i++)
		{
			cout << fdp[i] << " ";
		}
		cout << endl;

		imshow("input image", src);
		waitKey();
	}
}

void showHistogram(const string& name, int* hist, const int hist_cols,

	const int hist_height) {

	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
	// constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
		// colored in magenta

	}
	imshow(name, imgHist);
}

void testShowHistogram()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int histogram[256];
		calculateHistogram(src, histogram);

		imshow("input image", src);
		showHistogram("histogram", histogram, 256, 200);
		waitKey();
	}
}

void calculateReducedHistogram(Mat src, int histogram[], int size)
{
	
	memset(histogram, 0, 256 * sizeof(int));
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			histogram[src.at<uchar>(i, j)/size]++;
		}
	}
}

void testShowReducedHistogram()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		int histogram[256];
		calculateReducedHistogram(src, histogram, 16);
		for (int i = 0; i <src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				dst.at<uchar>(i, j) = src.at<uchar>(i, j) / 16;
				dst.at<uchar>(i, j) *= 16;

			}

		}
		imshow("input image", src);
		imshow("output image", dst);
		showHistogram("histogram", histogram, 256, 200);
		waitKey();
	}
}

void multipleSteps()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		float fdp[256];
		calculateFDP(src, fdp);
		const int WH = 5;
		const float TH = 0.0003f;

		vector<int> vf;
		for (int i = WH; i <= 255-WH; i++)
		{
			float medie = 0.0f;
			int max = 1; //true
			for (int j = 0; j < 2*WH+1; j++)
			{
				medie += fdp[i + j - WH];
				if (fdp[i+j-WH]> fdp[i])
				{
					max = 0; //false
				}
			}
			medie = medie / (2 * WH + 1);
			// && fdp[i] > medie + TH
			if (max && fdp[i] > medie + TH)
			{
				// rezulta ca i este maxim local
				vf.push_back(i);
			}
		}
		vf.push_back(255);
		vector<float> mijloace;
		for (int i = 0; i < vf.size()-1; i++)
		{
			mijloace.push_back((vf[i] + vf[i + 1]) / 2.0f);
		}
		mijloace.push_back(255);

		//segmentare
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar g = src.at<uchar>(i, j);
				float min = 256;
				int min_k = 256;
				for (int k = 0; k < mijloace.size(); k++)
				{
					if (abs(mijloace[k]-g)<min)
					{
						min = abs(mijloace[k] - g);
						min_k = k;
					}
				}
				dst.at<uchar>(i, j) = mijloace[min_k];
			}
		}
		imshow("input image", src);
		imshow("output image", dst);
		int hist_src[256] = { 0 };
		int hist_dst[256] = { 0 };
		calculateHistogram(src, hist_src);
		calculateHistogram(dst, hist_dst);
		showHistogram("histogram_src", hist_src, 256, 200);
		showHistogram("histogram_dst", hist_dst, 256, 200);
		waitKey();
	}
}


void dithering()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst_dithered = Mat(src.rows, src.cols, CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		float fdp[256];
		calculateFDP(src, fdp);
		const int WH = 5;
		const float TH = 0.0003f;

		vector<int> vf;
		for (int i = WH; i <= 255 - WH; i++)
		{
			float medie = 0.0f;
			int max = 1; //true
			for (int j = 0; j < 2 * WH + 1; j++)
			{
				medie += fdp[i + j - WH];
				if (fdp[i + j - WH] > fdp[i])
				{
					max = 0; //false
				}
			}
			medie = medie / (2 * WH + 1);
			// && fdp[i] > medie + TH
			if (max && fdp[i] > medie + TH)
			{
				// rezulta ca i este maxim local
				vf.push_back(i);
			}
		}
		vf.push_back(255);
		vector<float> mijloace;
		for (int i = 0; i < vf.size() - 1; i++)
		{
			mijloace.push_back((vf[i] + vf[i + 1]) / 2.0f);
		}
		mijloace.push_back(255);

		//segmentare
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar g = src.at<uchar>(i, j);
				float min = 256;
				int min_k = 256;
				for (int k = 0; k < mijloace.size(); k++)
				{
					if (abs(mijloace[k] - g) < min)
					{
						min = abs(mijloace[k] - g);
						min_k = k;
					}
				}
				dst.at<uchar>(i, j) = mijloace[min_k];
			}
		}

		//dithering
		
		int hist_src[256] = { 0 };
		int hist_dst[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar old_pixel = dst.at<uchar>(i,j);
				float min = 256.0f;
				int min_k = 0;
				for (int k = 0; k < 256; k++)
				{
					if (abs(fdp[k] - old_pixel) < min)
					{
						min = abs(fdp[k] - old_pixel);
						min_k = k;
					}
				}
				dst_dithered.at<uchar>(i, j) = hist_dst[min_k];
				float error = old_pixel - dst_dithered.at<uchar>(i, j);
				if (isInside(dst_dithered, i, j+1))
				{
					dst_dithered.at<uchar>(i, j) = dst_dithered.at<uchar>(i, j) + 7*error/16.0f;

				}
				if (isInside(dst_dithered, i+1, j - 1))
				{
					dst_dithered.at<uchar>(i+1, j-1) = dst_dithered.at<uchar>(i+1, j-1) + 3 * error / 16.0f;

				}
				if (isInside(dst_dithered, i+1, j))
				{
					dst_dithered.at<uchar>(i+1, j) = dst_dithered.at<uchar>(i+1, j) + 5 * error / 16.0f;

				}
				if (isInside(dst_dithered, i+1, j + 1))
				{
					dst_dithered.at<uchar>(i+1, j+1) = dst_dithered.at<uchar>(i+1, j+1) +  error / 16.0f;

				}
			}

		}
		

		imshow("input image", src);
		imshow("output image", dst);
		imshow("output image dithered", dst_dithered);
		
		calculateHistogram(src, hist_src);
		calculateHistogram(dst, hist_dst);
		showHistogram("histogram_src", hist_src, 256, 200);
		showHistogram("histogram_dst", hist_dst, 256, 200);
		waitKey();
	}
}
/*
void neighbor8(int* neighbors, Point p) 
{

}
*/
void bfs()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		
		ushort label = 0;
		Mat labels = Mat::zeros(src.size(),CV_16UC1);
		Mat dst = Mat::zeros(src.size(),CV_8UC3);
		Scalar colorLUT[1000] = { 0 };
		Scalar color;
		int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 }; // rand (coordonata orizontala)
		int di[8] = { 0, -1, -1, -1,  0,  1, 1, 1 }; // coloana (coordonata verticala)
		for (int i = 1; i < 1000; i++)
		{
			colorLUT[i] = Scalar(rand() & 255, rand() & 255, rand() & 255);
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i,j)==0 && labels.at<ushort>(i,j)==0)
				{
					label++;
					queue<Point> Q;
					labels.at<ushort>(i, j) = label;
					Q.push(Point(i, j));

					while (!Q.empty())
					{
						Point q = Q.front();
						int x = q.x;
						int y = q.y;
						Q.pop();
		

						uchar neighbors[8];
						
						for (int k = 0; k < 8; k++)
						{
							if (isInside(src, x+ dj[k], y + di[k]) )
							{
								if (src.at<uchar>(x + dj[k], y + di[k]) == 0 && labels.at<ushort>(x + dj[k], y + di[k]) == 0)
								{
									Q.push(Point(x + dj[k], y + di[k]));
									labels.at<ushort>( x + dj[k], y + di[k]) = label;
								}
								
							}
							
						}
						
					}

				}
			}

		}
		for (int i = 1; i < height-1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				//cout << labels.at<ushort>(i, j) << " ";
				Scalar color = colorLUT[labels.at<ushort>(i, j)];
				dst.at<Vec3b>(i, j)[0] = color[0];
				dst.at<Vec3b>(i, j)[1] = color[1];
				dst.at<Vec3b>(i, j)[2] = color[2];
			}
			//cout << endl;
		}

		cout << "here";
		imshow("input image", src);
		imshow("dst image", dst);
		waitKey();

	}

}


void twoPassings()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		ushort label = 0;
		Mat labels = Mat::zeros(src.size(), CV_16UC1);
		Mat dst = Mat::zeros(src.size(), CV_8UC3);
		vector<vector<int>> edges;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				/*if (src.at<uchar>(i, j) == 0 && labels<ushort>(i, j) == )
				{

				}*/
			}
		}

		
		imshow("input image", src);
		imshow("dst image", dst);
		waitKey();

	}

}


struct my_point
{
	my_point(int x, int y, byte c, char cd) : x(x), y(y), c(c), cd(cd)
	{};
	int x, y;
	byte c;
	char cd;
};


struct my_segment
{
	my_segment(my_point x, my_point y) : x(x), y(y)
	{};
	my_point x, y;

};
bool operator==(const my_point& lhs, const my_point& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;

}
void contur_tracing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		byte dir = 7;
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat::zeros(src.size(), CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		uchar OBIECT = 0;
		uchar FUNDAL = 255;
		vector<my_point> contur;
		bool finished = false;
		bool found = false;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i,j) == OBIECT)
				{
					dir = 7;
					contur.emplace_back(i, j, 7, 7 % 8);
					found = true;
					break;
				}
			}
			if (found == true)
			{
				break;
			}
		}
		int i = contur[0].x;
		int j = contur[0].y;
		cout << j << " " << i;
		int n = 0;
		int prev_dir = 7;
		while (!finished) {
			dir = dir % 2 == 0 ? dir = (dir + 7) % 8 : dir = (dir + 6) % 8;
			for (int k = 0; k < 8; k++) {
				int d = (dir + k) % 8;
				int x = i + di[d];
				int y = j + dj[d];
				if (isInside(src, x, y) && src.at<uchar>(x, y) == 0) {
					int prev_dir = contur.at(n).c;
					contur.emplace_back(x, y, d, (d - prev_dir));
					dir = d;
					i = x;
					j = y;
					n++;
					break;
				}
			}

			finished = n > 1 && contur[0] == contur[n - 1] && contur[1] == contur[n];
		}

		

		for (int i = 0; i < contur.size(); i++){
			dst.at<uchar>(contur[i].x, contur[i].y) = 255;
			cout <<i << " " << contur[i].x << " " << contur[i].y << " " << int(contur[i].c )<< " " << int(contur[i].cd) << endl;
		}
		imshow("src", src);
		imshow("dst", dst);


	}
}


vector<my_point>  get_contour(Mat src)
{
	
		//neigbhours
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		byte dir = 7;
		Mat dst = Mat::zeros(src.size(), CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		uchar OBIECT = 0;
		uchar FUNDAL = 255;
		vector<my_point> contur;
		bool finished = false;
		bool found = false;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				//finding the first pixel of the object
				if (src.at<uchar>(i, j) == OBIECT)
				{
					dir = 7;
					contur.emplace_back(j, i, 7, 7 % 8);
					found = true;
					break;
				}
			}
			if (found == true)
			{
				break;
			}
		}
		
		int i = contur[0].y;
		int j = contur[0].x;
		int n = 0;
		int prev_dir = 7;
		while (!finished) {

			dir = dir % 2 == 0 ? dir = (dir + 7) % 8 : dir = (dir + 6) % 8;
			//traversing the neighbours
			for (int k = 0; k < 8; k++) {
				int d = (dir + k) % 8;
				int y = i + di[d];
				int x = j + dj[d];
				//if it's contour
				if (isInside(src, x, y) && src.at<uchar>(y, x) == 0) {
					int prev_dir = contur.at(n).c;
					//adding the point 
					contur.emplace_back(x, y, d, (d - prev_dir));
					dir = d;
					i = y;
					j = x;
					n++;
					break;
				}
			}
			//end condition
			finished = n > 1 && contur[0] == contur[n - 1] && contur[1] == contur[n];
		}
		return contur;

}
//distance from P0 to P1P2
float distance(my_point P0, my_point P1, my_point P2)
{
	float num = abs((P2.y - P1.y) * P0.x - (P2.x - P1.x) * P0.y + P2.x * P1.y - P2.y * P1.x);
	float denum = sqrt((P2.y - P1.y) * (P2.y - P1.y) + (P2.x - P1.x) * (P2.x - P1.x));
	return num / denum;
}
//distance between two points
float distance(my_point P1, my_point P2) {
	return sqrt((P2.y - P1.y) * (P2.y - P1.y) + (P2.x - P1.x) * (P2.x - P1.x));

}
int getIndex(vector<my_point> v, my_point K)
{
	auto it = find(v.begin(), v.end(), K);

	// If element was found
	if (it != v.end())
	{

		// calculating the index of K
		int index = it - v.begin();
		return index;
	}
	else {
		return -1;
	}
}
void polygonal_aprox()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat::zeros(src.size(), CV_8UC1);
		//getting the contour
		vector<my_point> Q = get_contour(src);
	
		float epsilon = 10.0f;
		cout << "epsilon=";
		cin >> epsilon;
		for (int r = 0; r < 2; r++)
		{
			//drawing the contour in the other way
			if (r==1)
				reverse(Q.begin(), Q.end());
			deque<my_point> A;
			deque<my_point> B;

			//finding the farthest point from the first
			float max_dist = 0.0f;
			float d = 0.0f;
			int j = 0;
			for (int i = 1; i < Q.size(); i++)
			{
				d = distance(Q[i], Q[0]);

				if (d >= max_dist)
				{
					max_dist = d;
					j = i;

				}

			}

			//inserting the farthest point into A and B
			A.push_back(Q[j]);
			B.push_back(Q[j]);


			//inserting the first point into A
			A.push_back(Q[0]);
			bool ok = false;
			while (!A.empty())
			{
				int k = getIndex(Q, A.back());
				int l = getIndex(Q, B.back());

				int m = k;
				max_dist = 0.0f;
				d = 0.0f;
				//finding the farthest point on the contour between Pk and Pl from PkPl
				for (int i = k; i < l; i++)
				{
					d = distance(Q[i], Q[k], Q[l]);


					if (d > max_dist)
					{
						max_dist = d;
						m = i;
					}

				}
			
				//if the distance from the farthest point is bigger than epsilon we continue otherwise we move the last element from A to bo
				if (max_dist > epsilon)
				{
					A.push_back(Q[m]);
				}
				else
				{

					A.pop_back();
					B.push_back(Q[k]);
				}

			}
			//drawing the lines
			int n = B.size();
			for (int i = 0; i < n - 2; i++)
			{
				my_point X = B.front();
				B.pop_front();
				my_point Y = B.front();
				line(dst, Point(X.x, X.y), Point(Y.x, Y.y), Scalar(128, 128, 128), 2, LINE_8);
			}
			for (int i = 0; i < Q.size(); i++) {
				dst.at<uchar>(Q[i].y, Q[i].x) = 255;

			}
		}
		imshow("dst", dst);
		imshow("src", src);
		waitKey();
	}

}
void reconstruct() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		ifstream fin;
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		fin.open("reconstruct.txt");
		int x, y, n;
		fin >> x >> y >> n;
		int height = src.rows;
		int width = src.cols;
		src.at<uchar>(x,y) = 255;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		cout << n;
		for (int i = 0; i < n; i++)
		{
			int v;
			fin >> v;
			x = x + di[v];
			y = y + dj[v];
			src.at<uchar>(x, y) = 255;


		}
		imshow("excelent", src);
		cout << "here";
	}
}

#define FG 0 // obiect
#define BG 255 // fundal
void dilatare() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int i = 1; i < height-1; i++)
		{
			for (int j = 1; j < width-1; j++)
			{
				if (src.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = FG;
					}
				}

			}
		}
		imshow("src", src);
		imshow("dst", dst);
	}
}

void dilatareN() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int n = 1;
		cin >> n;
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat temp = src.clone();
		Mat dst = temp.clone();
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int l = 0; l < n; l++)
		{

			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (temp.at<uchar>(i, j) == FG)
					{
						for (int k = 0; k < 8; k++)
						{
							dst.at<uchar>(i + di[k], j + dj[k]) = FG;
						}
					}

				}
			}
			temp = dst.clone();
		}
		imshow("src", src);
		imshow("dst", dst);
	}
}

void eroziune() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (src.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{
						
						if (src.at<uchar>(i + di[k], j + dj[k]) == BG)
						{
							dst.at<uchar>(i , j ) = BG;
							break;
						}
					}
				}

			}
		}
		imshow("src", src);
		imshow("dst", dst);
	}
}

void eroziuneN() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat temp = src.clone();
		Mat dst = temp.clone();
		int n = 1;
		cin >> n;
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int l = 0; l < n; l++)
		{

			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (temp.at<uchar>(i, j) == FG)
					{
						for (int k = 0; k < 8; k++)
						{

							if (temp.at<uchar>(i + di[k], j + dj[k]) == BG)
							{
								dst.at<uchar>(i, j) = BG;
								break;
							}
						}
					}

				}
			}
			temp = dst.clone();
		}
		imshow("src", src);
		imshow("dst", dst);
	}
}

void inchidere() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat temp = src.clone();
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (src.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{
						temp.at<uchar>(i + di[k], j + dj[k]) = FG;
					}
				}

			}
		}
		Mat dst = temp.clone();
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (temp.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{

						if (temp.at<uchar>(i + di[k], j + dj[k]) == BG)
						{
							dst.at<uchar>(i, j) = BG;
							break;
						}
					}
				}

			}
		}
		imshow("src", src);
		imshow("dst", dst);
	}
}

void deschidere() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat temp = src.clone();
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (src.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{

						if (src.at<uchar>(i + di[k], j + dj[k]) == BG)
						{
							temp.at<uchar>(i, j) = BG;
							break;
						}
					}
				}

			}
		}
		Mat dst = temp.clone();
		
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (temp.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = FG;
					}
				}

			}
		}
		imshow("src", src);
		imshow("dst", dst);
	}
}

void extragereContur() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat temp = src.clone();
		int height = src.rows;
		int width = src.cols;
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int di[8] = { 0, -1, -1, -1,  0, 1, 1, 1 };
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (src.at<uchar>(i, j) == FG)
				{
					for (int k = 0; k < 8; k++)
					{

						if (src.at<uchar>(i + di[k], j + dj[k]) == BG)
						{
							temp.at<uchar>(i, j) = BG;
							break;
						}
					}
				}

			}
		}
		Mat dst = temp.clone();

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (temp.at<uchar>(i, j) == src.at<uchar>(i,j))
				{
					dst.at<uchar>(i, j) = BG;
				}
				else
				{
					dst.at<uchar>(i, j) = FG;

				}
			}
		}
		imshow("src", src);
		imshow("dst", dst);
	}

}

void calculateHistogramCumulative() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		int h[256];
		int hc[256];
		float f[256];
		float fc[256];

		calculateHistogram(src, h);
		calculateFDP(src, f);
		hc[0] = h[0];
		fc[0] = f[0];
		int sum = 0.0;
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < 256; i++)
		{
			hc[i] = hc[i - 1] + h[i];
			fc[i] = fc[i - 1] + f[i];
			sum += i * h[i];
		}
		int M = width * height;
		float mean = float(sum)/ M;

		float v = 0;
		for (int i = 0; i < 256; i++)
		{
			v += (i - mean)* (i - mean) * f[i];
		}
		cout << "Media" << " " << mean << endl;
		cout << "Std Dev" << " " << sqrt(v) ;
		imshow("input image", src);
		//imshow("output image", dst);
		showHistogram("histogram", hc, 256, 200);
		waitKey();
	}
	

}

void binarizareAutomataGlobala()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		dst = src.clone();
		int h[256];

		calculateHistogram(src, h);
		int height = src.rows;
		int width = src.cols;
		int M = width * height;
		float v = 0;
		int Imax = 0, Imin=0;
		for (int i = 0; i < 256; i++)
		{
			if (h[i])
			{
				Imax = i;
				if (Imin == 0)
				{
					Imin = i;
				}
			}
		}
		float T = (	 + Imax) / 2;
		float Tprev = T;
		float epsilon = 1.0f;
		do
		{
			int s1 = 0, n1 = 0;
			for (int g = Imin; g < T; g++)
			{
				s1 += g * h[g];
				n1 += h[g];
			}
			float uG1 = s1 / float(n1);

			int s2 = 0, n2 = 0;
			for (int g = T + 1; g < 256; g++)
			{
				s2 += g * h[g];
				n2 += h[g];
			}
			float uG2 = s2 / float(n2);
			Tprev = T;
			T = (uG1 + uG2) / 2;
			cout << T << endl;
		} while (abs(T-Tprev)> epsilon);
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i,j) < T)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;

				}
			}
		}

		imshow("input image", src);
		imshow("output image", dst);
		//imshow("output image", dst);
		showHistogram("histogram", h, 256, 200);
		waitKey();
	}
	
}

void histogramOperations() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst = src.clone();
	
		int h[256] = { 0 }, h2[256] = { 0 };
		calculateHistogram(src, h);
	
		int height = src.rows;
		int width = src.cols;
		
		int choice = 3;
		int gInMax = 0, gInMin = 0;
		int gOutMax = 255, gOutMin = 0;
		float gamma = 2.2;
		cin >> choice;
		switch (choice)
		{
		case 0:
			for (int i = 0; i < 256; i++)
			{
				h2[i] = 255 - h[i];
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					dst.at<uchar>(i, j) = 255 - src.at < uchar>(i, j);
				}
			}
			break;
		case 1:
			
			for (int i = 0; i < 256; i++)
			{
				if (h[i])
				{
					gInMax = i;	
					if (gInMin == 0)
					{
						gInMin = i;
					}
				}
			}
			for (int i = 0; i < 256; i++)
			{
				h2[i] = gOutMin + (h[i] - gInMin)*(gOutMax- gOutMin)/(gInMax - gInMin);
			}
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					dst.at<uchar>(i, j) = gOutMin + (src.at<uchar>(i, j) - gInMin) * (gOutMax - gOutMin) / (gInMax - gInMin);
				}
			}
			break;

		case 3:
			for (int i = 0; i < 256; i++)
			{
				h2[i] = 255*pow(h[i]/255,gamma);
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					dst.at<uchar>(i, j) = 255 * pow((src.at<uchar>(i,j) / 255.0f), gamma);
					if (dst.at<uchar>(i, j) <0)
					{
						dst.at<uchar>(i, j) = 0;
					}
					if (dst.at<uchar>(i, j) > 255)
					{
						dst.at<uchar>(i, j) = 255;
					}
				}
			}
			break;
		default:
			break;
		}
		
		imshow("input image", src);
		imshow("output image", dst);
		showHistogram("in", h, 256, 200);

		showHistogram("out", h2, 256, 200);
		waitKey();
	}


}

void egalizareHistograma() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		int h[256];
		int hc[256];

		calculateHistogram(src, h);
		hc[0] = h[0];
		int sum = 0.0;
		int height = src.rows;
		int width = src.cols;
		int tab[256] = { 0 };
		int M = height * width;
		for (int i = 0; i < 256; i++)
		{
			hc[i] = hc[i - 1] + h[i];
			tab[i] = 255 * hc[i] / M;
		}


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = tab[src.at<uchar>(i, j)];
			}
		}

		imshow("input image", src);

		imshow("output image", dst);
		showHistogram("histograma egalizata", tab, 256, 200);

		showHistogram("histogram", h, 256, 200);
		showHistogram("histograma cumulativa", hc, 256, 200);

		waitKey();
	}



}

enum FILTER_TYPE
{
	FTS,
	FTJ_INT,
	FTJ_FLT,

};

void convolutie(FILTER_TYPE filter_type) {
		char fname[MAX_PATH];
		while (openFileDlg(fname))
		{
			Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			Mat dst = Mat(src.rows, src.cols, CV_8UC1);


			int height = src.rows;
			int width = src.cols;
			int d = 1;
			int w = 3;
			cin >> w;
			d = w / 2;
			int laplace[3][3] = { { 0, -1 , 0}, {-1, 4, -1}, {0,-1, 0} };

			int median[7][7] = { { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 } };
			int H[7][7] = { 0 };
			switch (filter_type)
			{
			case FTS:
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						H[i][j] = laplace[i][j];
					}
				}
				break;
			case FTJ_INT:
				for (int i = 0; i < 7; i++)
				{
					for (int j = 0; j < 7; j++)
					{
						H[i][j] = median[i][j];
					}
				}
				break;
			case FTJ_FLT:
				break;
			default:
				break;
			}
			//int H[7][7] = { { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 }, { 1,1,1,1,1,1,1 } };

		
		//FTJ
		int fs=0;
		for (int y = 0; y <= 2*d; y++)
		{
			for (int x = 0; x <= 2*d; x++)
			{
				fs += H[y][x];
			}
		}
		//FTS
		int sp = 0;
		int sn = 0;
		for (int y = 0; y <= 2*d; y++)
		{
			for (int x = 0; x <= 2 * d; x++)
			{
				if (H[y][x] < 0)
					sn += -H[y][x];
				else
				{
					sp += H[y][x];
				}
			}
		}
		for (int i = d; i < height-d; i++)
		{
			for (int j = d; j < width - d; j++)
			{
				float sum = 0;
				for (int y = -d; y <= d; y++)
				{
					for (int x = -d; x <= d; x++)
					{
						sum += H[y + d][x + d] * src.at<uchar>(i+y,x+j);

					}
				}

				switch (filter_type)
				{
				case FTS:
					fs = 2 * max(sp, sn);
					dst.at<uchar>(i, j) = sum / fs +128;

					break;
				case FTJ_INT:
					dst.at<uchar>(i, j) = sum / fs;

					break;
				case FTJ_FLT:
					break;
				default:
					break;
				}


			}
		}

		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}

}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src, int option)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	// Centering transformation 
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // chanels[0] = Re(DFT(I), chanels[1] = Im(DFT(I))

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	Mat magr,phir;
	// Dislplay here the phase and magnitude
	// ......
	mag += Scalar::all(1);
	log(mag, mag);
	normalize(mag, magr, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(phi, phir, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("magnitudine", magr);
	imshow("phaase", phir);
	// Insert filtering operations here ( chanles[0] = Re(DFT(I), chanels[1] = Im(DFT(I) )
	int poz;
	float coef;
	float R = 20; // filter "radius"
	int height = src.rows;
	int width = src.cols;
	float opRez;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			switch (option) {
			case 1: break; // NO filter
			case 2:
				// FTJ ideal
				// inserati codul de filtrare aici ...
				opRez = pow((height / 2 - i), 2) + pow((width / 2 - j), 2);
				if (opRez > pow(R, 2)) {
					opRez = 0;
					channels[0].at<float>(i, j) *= opRez;
					channels[1].at<float>(i, j) *= opRez;


				}
		
				break;
			case 3:
				// FTS ideal
				// inserati codul de filtrare aici ...
				opRez = pow((height / 2 - i), 2) + pow((width / 2 - j), 2);
				if (opRez <= pow(R, 2)) {
					opRez = 0;
					channels[0].at<float>(i, j) *= opRez;
					channels[1].at<float>(i, j) *= opRez;
				}
			
				break;
			case 4:
				// FTJ gauss
				// inserati codul de filtrare aici ...
				opRez= exp(-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2)) / pow(R, 2));
				channels[0].at<float>(i, j) *= opRez;
				channels[1].at<float>(i, j) *= opRez;
				break;
			case 5:
				// FTJ gauss
				// inserati codul de filtrare aici ...
				opRez = 1 - exp(-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2)) / pow(R, 2));
				channels[0].at<float>(i, j) *= opRez;
				channels[1].at<float>(i, j) *= opRez;
				break;
			}
		}
	}
	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	centering_transform(dstf);
	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//dstf.convertTo(dst, CV_8UC1);

	return dst;
}

void dft()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	
		Mat dst1 = generic_frequency_domain_filter(src, 2);
		Mat dst2 = generic_frequency_domain_filter(src, 3);
		Mat dst3 = generic_frequency_domain_filter(src, 4);
		Mat dst4 = generic_frequency_domain_filter(src, 5);



		imshow("input image", src);
		imshow("FTJ ideal", dst1);
		imshow("FTS ideal", dst2);
		imshow("FTJ Gauss", dst3);
		imshow("FTS Gauss", dst4);
		waitKey();

	}
}

void testLab10FiltruGaussian1() {
	Mat src, dst[5];
	char* nume[] = { "Output 3", "Output 5", "Output 7", "Output 9" };
	char fname[MAX_PATH] = "cameraman.bmp";
	float H[9][9] = {};
	float fs = 0;

	if (1 || openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("Input", src);
		for (int d = 1; d <= 4; d++) {
			int w = 2 * d + 1;
			fs = 0;
			dst[d] = src.clone();
			float zgomot = w / 6.0f;
			float E, M = 2 * PI * zgomot * zgomot;
			for (int y = 0; y < w; y++)
				for (int x = 0; x < w; x++) {
					E = exp(-(float)((x - d) * (x - d) + (y - d) * (y - d)) / (2 * zgomot * zgomot));
					H[y][x] = E / M;
					fs += H[y][x];
				}
			printf("Sum = %f\n", fs);
			auto begin = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					float sum = 0;
					for (int y = -d; y <= d; y++)
						for (int x = -d; x <= d; x++) {
							if (y + i >= 0 && y + i <= height - 1 && x + j >= 0 && x + j <= width - 1) {
								sum += H[y + d][x + d] * src.at<uchar>(i + y, x + j);
							}
						}
					dst[d].at<uchar>(i, j) = sum / fs;
				}
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
			printf("Time measured: %.6f seconds for %s.\n", elapsed.count() * 1e-9, nume[d - 1]);
			imshow(nume[d - 1], dst[d]);
		}
	}
}

void testLab10FiltruGaussian2() {
	Mat src, dst[5];
	char* nume[] = { "Output 3 2", "Output 5 2", "Output 7 2", "Output 9 2" };
	char fname[MAX_PATH] = "cameraman.bmp";
	float H[9] = {};
	float fs = 0;

	if (1 || openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("Input", src);
		for (int d = 1; d <= 4; d++) {
			int w = 2 * d + 1;
			fs = 0;
			dst[d] = src.clone();
			Mat temp = src.clone();
			float zgomot = w / 6.0f;
			float E, M = sqrt(2 * PI) * zgomot;
			for (int y = 0; y < w; y++)
			{
				E = exp(-(float)((y - d) * (y - d)) / (2 * zgomot * zgomot));
				H[y] = E / M;
				fs += H[y];
			}
			printf("Sum = %f\n", fs);
			auto begin = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					float sum = 0;
					for (int y = -d; y <= d; y++)
						if (y + i >= 0 && y + i <= height - 1) {
							sum += H[y + d] * src.at<uchar>(i + y, j);
						}
					temp.at<uchar>(i, j) = sum / fs;
				}
			}
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					float sum = 0;
					for (int x = -d; x <= d; x++)
						if (x + j >= 0 && x + j <= width - 1) {
							sum += H[x + d] * temp.at<uchar>(i, j + x);
						}
					dst[d].at<uchar>(i, j) = sum / fs;
				}
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
			printf("Time measured: %.6f seconds for %s.\n", elapsed.count() * 1e-9, nume[d - 1]);
			imshow(nume[d - 1], dst[d]);
		}

	}
}



int neighboursCoordsX[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
int neighboursCoordsY[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

int laplaceFilter[] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
int highPassFilter[] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
Mat applyFilterInSpacialDomain(Mat img, int filter[]) {


	Mat rezult = img.clone();

	float scalingFactor = 0;
	float highPassAdditionFactor = 0;

	float pozitiveElementsSum = 0;
	float negativeElementsSum = 0;

	for (int i = 0; i < 9; ++i) {
		if (filter[i] > 0) {
			pozitiveElementsSum += filter[i];
		}
		else {
			negativeElementsSum += abs(filter[i]);
		}
	}

	//low pass filter
	if (negativeElementsSum == 0) {
		scalingFactor = 1 / pozitiveElementsSum;
	}
	else { //high pass filter
		scalingFactor = 1 / (2 * max(pozitiveElementsSum, negativeElementsSum));
		highPassAdditionFactor = 127;
	}

	for (int i = 1; i < img.rows - 1; ++i) {
		for (int j = 1; j < img.cols - 1; ++j) {
			float newPixelValue = 0;

			for (int k = 0; k < 9; ++k) {
				int x = i + neighboursCoordsX[k];
				int y = j + neighboursCoordsY[k];

				newPixelValue = newPixelValue + img.at<uchar>(x, y) * filter[k];
			}

			newPixelValue = newPixelValue * scalingFactor + highPassAdditionFactor;

			if (newPixelValue < 0) {
				newPixelValue = 0;
			}

			if (newPixelValue > 255) {
				newPixelValue = 255;
			}

			rezult.at<uchar>(i, j) = newPixelValue;
		}
	}

	return rezult;
}

// filtru median 
void swap(uchar* a, uchar* b)
{
	byte t = *a;
	*a = *b;
	*b = t;
}

int partition(uchar arr[], int low, int high)
{
	uchar pivot = arr[high]; // pivot
	int i = (low - 1);

	for (int j = low; j <= high - 1; j++)
	{
		if (arr[j] < pivot)
		{
			i++;
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}
void quickSort(uchar arr[], int low, int high)
{
	if (low < high)
	{
		int pi = partition(arr, low, high);
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}


Mat medianFilter(Mat src) {
	int height = src.rows;
	int width = src.cols;
	int w = 5;
	printf("Write w=(3,5,7,9):");
	scanf("%d", &w);
	int d = 1;
	d = w / 2;
	Mat dst = src.clone();
	double t = (double)getTickCount();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			uchar* L = (uchar*)malloc(w * w);
			if (L == NULL) {
				printf("Nu s-a putut aloca memorie\n");
				exit(1);
			}
			int contor = 0;
			for (int m = -d; m <= d; m++) {
				for (int n = -d; n <= d; n++) {
					L[contor] = src.at<uchar>(i + m, j + n);
					contor++;
				}
			}
			quickSort(L, 0, w * w - 1);
			dst.at<uchar>(i, j) = L[w * w / 2];
			free(L);
		}

	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void testMedianFiltre() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)getTickCount();
	Mat dst = medianFilter(src);
	imshow("Imagine sursa:", src);
	imshow("Imagine destinatie", dst);
	waitKey(0);
}

//Zgomit Gaussian
Mat getConvolutie(Mat src, int d, float fs, float S[9][9]) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			float sum = 0.0;
			for (int y = -d; y <= d; y++) {
				for (int x = -d; x <= d; x++) {
					sum += (S[y + d][x + d] * src.at<uchar>(i + y, j + x));
				}
			}
			dst.at<uchar>(i, j) = (uchar)(sum / fs);
		}
	}
	return dst;
}


Mat gaussianFiltre(Mat src) {
	int height = src.rows;
	int width = src.cols;
	int w = 3;
	printf("Write w=(3,5,7):");
	scanf("%d", &w);
	int d = 1;
	d = w / 2;
	float S[9][9];
	float sum = 0.0f;
	float sigma = ((float)w) / 6.0f;
	Mat dst;
	for (int y = 0; y < w; y++) {
		for (int x = 0; x < w; x++) {
			float E = exp(-(x - d) * (x - d) + (y - d) * (y - d) / (2 * sigma * sigma));
			float N = 2 * PI * sigma * sigma;
			S[y][x] = E / N;
			sum += S[y][x];
		}
	}
	printf("Suma %f\n", sum);
	//convlolutia
	double t = (double)getTickCount();
	dst = getConvolutie(src, d, sum, S);
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void testGaussianFiltre() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)getTickCount();
	Mat dst = gaussianFiltre(src);
	imshow("Imagine sursa:", src);
	imshow("Imagine destinatie", dst);
	waitKey(0);
}

Mat optimizedGaussian(Mat src) {
	int height = src.rows;
	int width = src.cols;
	float sum = 0.0f;
	int w = 3;
	printf("Write w=(3,5,7):");
	scanf("%d", &w);
	int d = 1;
	d = w / 2;
	float S[9];
	float sigma = ((float)w) / 6.0f;
	for (int x = 0; x < w; x++) {
		float E = exp(-(x - d) * (x - d) / (2 * sigma * sigma));
		float N = sigma * sqrt(2 * PI);
		S[x] = E / N;
		sum += S[x];
	}
	printf("Sum : %f\n", sum);
	Mat temp = src.clone();
	double t = (double)getTickCount();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			float ps = 0.0f;
			for (int m = -d; m <= d; m++) {
				ps += src.at<uchar>(i + m, j) * S[m + d];
			}
			temp.at<uchar>(i, j) = ps / sum;
		}
	}
	//pentru destinatie
	Mat dst = src.clone();
	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			float ps = 0.0f;
			for (int m = -d; m <= d; m++) {
				ps += temp.at<uchar>(i, j + m) * S[m + d];
			}
			dst.at<uchar>(i, j) = ps / sum;
		}
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void testOptimizedGaussianFiltre() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	double t = (double)getTickCount();
	Mat dst = optimizedGaussian(src);
	imshow("Imagine sursa:", src);
	imshow("Imagine destinatie", dst);
	waitKey(0);
}

void canny()
{
	char fname[MAX_PATH];
	//int Sx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	//int Sy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	int Sx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int Sy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	int H[9] = { 0 };

	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat temp = src.clone();
		Mat temp2 = src.clone();
		Mat modul = Mat::zeros(src.size(), CV_8UC1);
		Mat directie = Mat::zeros(src.size(), CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		//gauss
		//temp = applyFilterInSpacialDomain(src, gaussianFilter);
		/*
		int w = 5;
		int d = (w - 1) / 2;
		int fs = 0;
		float zgomot = w / 6.0f;
		float E, M = sqrt(2 * PI) * zgomot;
		for (int y = 0; y < w; y++)
		{
			E = exp(-(float)((y - d) * (y - d)) / (2 * zgomot * zgomot));
			H[y] = E / M;
			fs += H[y];
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float sum = 0;
				for (int y = -d; y <= d; y++)
					if (y + i >= 0 && y + i <= height - 1) {
						sum += H[y + d] * src.at<uchar>(i + y, j);
					}
				temp2.at<uchar>(i, j) = sum / fs;
			}
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float sum = 0;
				for (int x = -d; x <= d; x++)
					if (x + j >= 0 && x + j <= width - 1) {
						sum += H[x + d] * temp2.at<uchar>(i, j + x);
					}
				temp.at<uchar>(i, j) = sum / fs;
			}
		}
		*/
		
		
		//Mat gradX = applyFilterInSpacialDomain(temp, Sx);
		//Mat gradY = applyFilterInSpacialDomain(temp, Sy);

		int gradX = 0;
		int gradY = 0;

		for (int i = 0; i < height; i++)
		{
			for (int j = 1; j < width; j++)
			{
				int k = 0;
				gradX = 0;
				gradY = 0;
				for (int x = -1; x <= 1; x++)
				{
					for (int y = -1; y <= 1; y++)
					{

						gradX+= Sx[x + 1][y + 1] * temp.at<uchar>(i + x, j + y);
						gradY += Sy[x + 1][y + 1] * temp.at<uchar>(i + x, j + y);
					}
				}
				modul.at<uchar>(i, j) = sqrt(gradX * gradX + gradY * gradY )/5.65f;

				float teta = atan2(float(gradX), float(gradY));
				/*
				int dir = 0;
				if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
				if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
				if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
				if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) dir = 3;
				directie.at<uchar>(i, j) = dir;
				*/
				//cout << i << " " << j <<endl;
			}
		}
	
		

		
		imshow("Src", src);
		imshow("Modul", modul);

	}
}



#define WEAK 128
#define STRONG 255
void canny2()
{
	Mat src, dst;
	char fname[MAX_PATH];
	float H[9] = {};
	float fs = 0;
	int Sx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int Sy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	if ( openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat temp = src.clone();
		Mat modul = Mat::zeros(src.size(), CV_8UC1);
		Mat directie = Mat::zeros(src.size(), CV_8UC1);

		int height = src.rows;
		int width = src.cols;
		imshow("Sursa", src);

		// Filtru gaussian
		int w = 3;
		//cout << "w = ";
		//cin >> w;
		int d = (w - 1) / 2;

		fs = 0;
		dst = src.clone();
		float zgomot = w / 6.0f;
		float E, M = sqrt(2 * PI) * zgomot;
		for (int y = 0; y < w; y++)
		{
			E = exp(-(float)((y - d) * (y - d)) / (2 * zgomot * zgomot));
			H[y] = E / M;
			fs += H[y];
		}
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 1; i < height -1; i++) {
			for (int j = 1; j < width -1; j++) {
				float sum = 0;
				for (int y = -d; y <= d; y++)
					if (y + i >= 0 && y + i <= height - 1) {
						sum += H[y + d] * src.at<uchar>(i + y, j);
					}
				temp.at<uchar>(i, j) = sum / fs;
			}
		}
		for (int i = 1; i < height -1; i++) {
			for (int j = -1; j < width -1; j++) {
				float sum = 0;
				for (int x = -d; x <= d; x++)
					if (x + j >= 0 && x + j <= width - 1) {
						sum += H[x + d] * temp.at<uchar>(i, j + x);
					}
				dst.at<uchar>(i, j) = sum / fs;
			}
		}
		
		imshow("Gaussian", dst);
		// calcul modul si directie
		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {
				float sum = 0;
				int gradX = 0, gradY = 0;
				for (int x = -1; x <= 1; x++)
					for (int y = -1; y <= 1; y++) {
						gradX += Sx[x + 1][y + 1] * dst.at<uchar>(i + x, j + y);
						gradY += Sy[x + 1][y + 1] * dst.at<uchar>(i + x, j + y);
					}
				modul.at<uchar>(i, j) = sqrt(gradX * gradX + gradY * gradY) / 5.65f; // 4 * sqrt(2)
				int dir = 0;
				float teta = atan2((float)gradY, (float)gradX);
				if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
				if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
				if ((teta > -PI / 8 && teta < PI / 8) || (teta > 7 * PI / 8 && teta < -7 * PI / 8)) dir = 2;
				if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) dir = 3;
				directie.at<uchar>(i, j) = dir;
			}
		Mat modul2 = modul.clone();
		imshow("Modul", modul);
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				switch (directie.at<uchar>(i,j))
				{
				case 0:
					if (modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j) || modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j))
						modul2.at<uchar>(i, j) = 0;
					break;
				case 2:
					if (modul.at<uchar>(i, j) < modul.at<uchar>(i , j -1) || modul.at<uchar>(i, j) < modul.at<uchar>(i , j + 1) )
						modul2.at<uchar>(i, j) = 0;
					break;
				case 1:
					if (modul.at<uchar>(i, j) < modul.at<uchar>(i-1, j - 1) || modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j +1))
						modul2.at<uchar>(i, j) = 0;
					break;
				case 3:
					if (modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j + 1) || modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j - 1))
						modul2.at<uchar>(i, j) = 0;
					break;
				default:
					break;
				}
			}
		}
		imshow("Modul NMS", modul2);

		int hist[256] = { 0 };
		float p = 0.1;
		float k = 0.4;
		//calculateHistogram(modul2, hist);
		for (int i = 1; i < height-1; i++)
		{
			for (int j = 1; j < width-1; j++)
			{
				hist[modul2.at<uchar>(i, j)]++;
			}
		}
		int nrNonMuchie = (1 - p) * ((height - 2) * (width - 2) - hist[0]);
		int sum = 0;
		int i;
		for (i = 1; i < 256; i++)
		{
			sum += hist[i];
			if (sum> nrNonMuchie)
			{
				break;
			}
		}

		int pH = i;
		int pL = k * pH;
		cout << pH << " " << pL << endl;
		Mat modul3 = modul2.clone();
		Mat modul4 = modul2.clone();
		for (int i = 1; i < height -1; i++)
		{
			for (int j = 1; j < width -1; j++)
			{
				//if (modul2.at<uchar>(i, j) < pL)
					//modul3.at<uchar>(i, j) = 0;
				if (modul2.at<uchar>(i, j) > 20)
					modul3.at<uchar>(i, j) = STRONG;
				else
					modul3.at<uchar>(i, j) = 0;

				/*
				if (pL < modul2.at<uchar>(i, j) && modul2.at<uchar>(i, j) < pH)
				{
					modul3.at<uchar>(i, j) = WEAK;

				}
				*/
			}
		}
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (modul2.at<uchar>(i, j) < pL)
					modul4.at<uchar>(i, j) = 0;
				if (modul2.at<uchar>(i, j) > pH)
					modul4.at<uchar>(i, j) = STRONG;

				
				if (pL < modul2.at<uchar>(i, j) && modul2.at<uchar>(i, j) < pH)
				{
					modul4.at<uchar>(i, j) = WEAK;

				}
				
			}
		}
		
		queue<Point> que;
		int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 }; // rand (coordonata orizontala)
		int di[8] = { 0, -1, -1, -1,  0,  1, 1, 1 }; // coloana (coordonata verticala)
		Mat visited = Mat::zeros(src.size(), CV_8UC1);
		Mat modul5 = modul4.clone();
		for (int i = 2; i < height - 3 ; i++)
		{
			for (int j = 2; j < width - 3; j++)
			{
				if (modul5.at<uchar>(i,j) == STRONG && visited.at<uchar>(i,j) == 0)
				{
					que.push(Point(j, i));
					visited.at<uchar>(i, j) = 1;
				}
				while (!que.empty())
				{
					Point oldest = que.front();
					int jj = oldest.x;
					int ii = oldest.y;
					que.pop();

					for (int k = 0; k < 8; k++)
					{
						if (modul5.at<uchar>(ii + di[k],jj + dj[k]) == WEAK)
						{
							modul5.at<uchar>(ii + di[k], jj + dj[k]) = STRONG;
							visited.at<uchar>(ii + di[k], jj + dj[k]) = 1;
							que.push(Point(ii + di[k], jj + dj[k]));
						}

					}

				}
			}
		}

		for (int i = 1; i < height -1 ; i++)
		{
			for (int j = 1; j < height - 1; j++)
			{
				if (modul5.at<uchar>(i,j) == WEAK)
				{
					modul5.at<uchar>(i, j) = 0;
				}

			}
		}
		Mat ImgDir = Mat::zeros(src.size(), CV_8UC3);
		Scalar colorLut[4] = { 0 };
		colorLut[0] = Scalar(0, 0, 255);
		colorLut[1] = Scalar(0, 255, 255);
		colorLut[2] = Scalar(255, 0, 0);
		colorLut[3] = Scalar(0, 255, 0);
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (modul5.at<uchar>(i, j))
				{
					Scalar color = colorLut[directie.at<uchar>(i, j)];
					ImgDir.at<Vec3b>(i, j)[0] = color[0];
					ImgDir.at<Vec3b>(i, j)[1] = color[1];
					ImgDir.at<Vec3b>(i, j)[2] = color[2];

				}
			}
		}

		imshow("modul binarizatx`", modul3);
		imshow("modul weak/strong", modul4);
		imshow("modul extindere muchii", modul5);
		imshow("directii", ImgDir);
		waitKey();

		
	}

}

Mat gaussianFiltre(Mat src, int w) {
	int height = src.rows;
	int width = src.cols;
	int d = 1;
	d = w / 2;
	float S[9][9];
	float sum = 0.0f;
	float sigma = ((float)w) / 6.0f;
	Mat dst;
	for (int y = 0; y < w; y++) {
		for (int x = 0; x < w; x++) {
			float E = exp(-(x - d) * (x - d) + (y - d) * (y - d) / (2 * sigma * sigma));
			float N = 2 * PI * sigma * sigma;
			S[y][x] = E / N;
			sum += S[y][x];
		}
	}
	printf("Suma %f\n", sum);
	//convlolutia
	double t = (double)getTickCount();
	dst = getConvolutie(src, d, sum, S);
	t = ((double)getTickCount() - t) / getTickFrequency();
	//Print the proccessing time
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

void computeModulAndDirection(Mat temp, Mat* modul, Mat* directie, int d) {
	int Sx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int Sy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
	int height = temp.rows;
	int width = temp.cols;


	for (int i = d; i < height - d; i++) {
		for (int j = d; j < width - d; j++) {
			int gradX = 0;
			int gradY = 0;
			for (int y = -d; y <= d; y++) {
				for (int x = -d; x <= d; x++) {
					gradX += (Sx[y + d][x + d] * temp.at<uchar>(i + y, j + x));
					gradY += (Sy[y + d][x + d] * temp.at<uchar>(i + y, j + x));
				}
			}
			(*modul).at<uchar>(i, j) = sqrt(gradX * gradX + gradY * gradY) / 5.65;
			int dir = 0;
			float teta = atan2((float)gradY, (float)gradX);
			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8))
				dir = 0;

			if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8))
				dir = 1;

			if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8)
				dir = 2;

			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8))
				dir = 3;

			(*directie).at<uchar>(i, j) = dir;
		}
	}


}

void nonMaximumSuppresion(Mat* modul, Mat direction) {
	int height = modul->rows;
	int width = modul->cols;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			switch (direction.at<uchar>(i, j)) {
			case 0:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i - 1, j) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i + 1, j))
					modul->at<uchar>(i, j) = 0;
				break;
			case 1:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i - 1, j - 1) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i + 1, j + 1))
					modul->at<uchar>(i, j) = 0;
				break;
			case 2:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i, j - 1) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i, j + 1))
					modul->at<uchar>(i, j) = 0;
				break;
			case 3:
				if (modul->at<uchar>(i, j) < modul->at<uchar>(i - 1, j + 1) ||
					modul->at<uchar>(i, j) < modul->at<uchar>(i + 1, j - 1))
					modul->at<uchar>(i, j) = 0;
				break;
			}
		}
	}
}
#define WEAK 128 
#define STRONG 255 
void compute_histogram2(int* histogram, int n, Mat modul) {
	int height = modul.rows;
	int width = modul.cols;
	for (int i = 0; i < n; i++) {
		histogram[i] = 0;
	}
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			histogram[modul.at<uchar>(i, j)] += 1;
		}
	}
}
void computeAdaptiveThreshold(Mat modul, int* histogram, int* pH, int* pL) {
	float p = 0.1f;
	float k = 0.4f;
	int height = modul.rows;
	int width = modul.cols;
	compute_histogram2(histogram, 256, modul);
	//calculare nr pixeli cu modul diferit de zero care nu vor fi puncte de muchie
	float nrNonMuchie = (1 - p) * ((height - 2) * (width - 2) - histogram[0]);
	//Calculati Pragul adaptiv insumand elementele din Hist(incepand de la Hist[1]),
	//oprindu - va in momentul in care suma > NrNonMuchie.
	float sum = 0.0f;
	int adaptiveThreshold = 0;
	for (int i = 1; i < 256; i++) {
		sum += histogram[i];
		if (sum > nrNonMuchie) {
			adaptiveThreshold = i;
			break;
		}
	}
	*pH = adaptiveThreshold;
	*pL = k * adaptiveThreshold;
}

void histerezaBinarize(Mat* modul, int pl, int ph) {
	int height = modul->rows;
	int width = modul->cols;
	for (int i = 1; i < height -1; i++) {
		for (int j = 1; j < width -1; j++) {
			if (modul->at<uchar>(i, j) < pl)
				modul->at<uchar>(i, j) = 0;
			else if (modul->at<uchar>(i, j) > ph)
				modul->at<uchar>(i, j) = STRONG;
			else
				modul->at<uchar>(i, j) = WEAK;
		}
	}
}

void edgesExtension(Mat* modul) {
	Mat_<uchar> visited = Mat::zeros(modul->size(), CV_8UC1);
	queue<Point> que;
	int height = modul->rows;
	int width = modul->cols;
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 }; // row 
	int di[8] = { 0, -1, -1, -1,  0,  1, 1, 1 }; // col 

	for (int i = 2; i < height - 3; i++) {
		for (int j = 2; j < width - 3; j++) {
			if (modul->at<uchar>(i, j) == STRONG && visited(i, j) == 0) {
				que.push(Point(j, i));
				visited(i, j) = 1;
			}
			while (!que.empty()) {
				Point oldest = que.front();
				int jj = oldest.x;
				int ii = oldest.y;
				que.pop();
				int mag = 0;
				for (int d = 0; d < 8; d++) {
					if (modul->at<uchar>(ii + di[d], jj + dj[d]) == WEAK) {
						modul->at<uchar>(ii + di[d], jj + dj[d]) = STRONG;
						que.push(Point(jj + dj[d], ii + di[d]));
						visited(ii + di[d], jj + dj[d]) = 1;
					}
				}
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (modul->at<uchar>(i, j) == WEAK) {
				modul->at<uchar>(i, j) = 0;
			}
		}
	}

}
void showDirections(Mat modul, Mat directie) {
	Scalar colorLUT[4] = { 0 };
	colorLUT[0] = Scalar(0, 0, 255); //red 
	colorLUT[1] = Scalar(0, 255, 255); // yellow 
	colorLUT[2] = Scalar(255, 0, 0); // blue 
	colorLUT[3] = Scalar(0, 255, 0); // green 
	Mat_<Vec3b> ImgDir = Mat::zeros(modul.size(), CV_8UC3);
	int d = 1;
	int height = modul.rows;
	int width = modul.cols;
	for (int i = d; i < height - d; i++) // d=1 
		for (int j = d; j < width - d; j++)
			if (modul.at<uchar>(i, j)) {
				Scalar color = colorLUT[directie.at<uchar>(i, j)];
				ImgDir(i, j)[0] = color[0];
				ImgDir(i, j)[1] = color[1];
				ImgDir(i, j)[2] = color[2];
			}
	imshow("Directii", ImgDir);
}
void  cannyFunction(Mat src) {
	Mat temp = src.clone();
	Mat modul = Mat::zeros(src.size(), CV_8UC1);
	Mat directie = Mat::zeros(src.size(), CV_8UC1);
	int w = 3;
	int d = w / 2;
	temp = gaussianFiltre(src, w);

	computeModulAndDirection(temp, &modul, &directie, d);
	imshow("Modul raw", modul);
	nonMaximumSuppresion(&modul, directie);
	imshow("Modu after NMS", modul);
	//compute adaptive threshold 
	int histogram[256];
	//stabilire praguri de binarizare
	int pH, pL;
	computeAdaptiveThreshold(modul, histogram, &pH, &pL);
	//binarizare cu histereza
	printf("PH %d", pH);
	printf("P%d", pL);

	histerezaBinarize(&modul, pL, pH);
	imshow("Modul binarizare histereza", modul);
	//extinderea muchiilor
	edgesExtension(&modul);
	imshow("Modul extindere muchii", modul);

	//afisare cu cod de culoari
	showDirections(modul, directie);
	//showHistogram("Histogram", histogram, 256, 300);
	waitKey(0);
}
void testCannyFunction() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	cannyFunction(src);
}
void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

int AriaCentruMasa(const Mat_<Vec3b>& img, const Vec3b& obj_color, int* ri, int* ci)
{
	int area = 0;
	int rr = 0;
	int cc = 0;
	int i = 0;
	int j = 0;
	int aux = 0;
	int height = img.rows;
	int weight = img.cols;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < weight; j++)
		{
			if (img(i, j) == obj_color)
			{
				area = area + 1;
				rr = rr + i;
				cc = cc + j;
			}
		}
	}
	aux = rr / area;
	(*ri) = aux;
	aux = cc / area;
	(*ci) = aux;

	return area;
}




void AxaAlungire(Mat_<Vec3b> img, Vec3b col, int ri, int ci, double* angle)
{
	float num = 0;
	float f1 = 0;
	float f2 = 0;
	int i = 0;
	int j = 0;
	int height = img.rows;
	int weight = img.cols;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < weight; j++)
		{
			if (img(i, j) == col)
			{
				num = num + (2 * (i - ri) * (j - ci));
				f1 = f1 + ((j - ci) * (j - ci));
				f2 = f2 + ((i - ri) * (i - ri));
			}
		}
	}

	*(angle) = atan2(num, f1 - f2) / 2;
}

void Perimeter(const Mat_<Vec3b>& img, const Vec3b& obj_color, int* perimeter, Mat_<Vec3b>& img2)
{
	int i = 0;
	int j = 0;
	int height = img.rows;
	int width = img.cols;
	img2 = img.clone();

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (img(i, j) == obj_color && (
				(isInside(img, i, j + 1) && img(i, j + 1) != obj_color)
				|| (isInside(img, i, j - 1) && img(i, j - 1) != obj_color)
				|| (isInside(img, i + 1, j) && img(i + 1, j) != obj_color)
				|| (isInside(img, i - 1, j) && img(i - 1, j) != obj_color)
				|| (isInside(img, i + 1, j + 1) && img(i + 1, j + 1) != obj_color)
				|| (isInside(img, i - 1, j + 1) && img(i - 1, j + 1) != obj_color)
				|| (isInside(img, i + 1, j - 1) && img(i + 1, j - 1) != obj_color)
				|| (isInside(img, i - 1, j - 1) && img(i - 1, j - 1) != obj_color)))
			{
				(*perimeter)++;
				img2(i, j) = Vec3b(0, 0, 0);
			}
		}
	}

	*perimeter = *perimeter ;


}

void FactorSubtiere(int area, int perimeter, double& par)
{
	int pp = perimeter * perimeter;
	par = 4 * PI * ((double)area / (pp));
}

void FactorAspect(const Mat_<Vec3b>& img, const Vec3b& obj_color, double& par2)
{
	int i = 0;
	int j = 0;
	int height = img.rows;
	int weight = img.cols;
	int cmax = 0;
	int rmax = 0;
	int	cmin = 0x7FFFFFFF;
	int rmin = 0x7FFFFFFF;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < weight; j++)
		{
			if (img(i, j) == obj_color)
			{
				if (rmin > i)
				{
					rmin = i;
				}
				if (rmax < i)
				{
					rmax = i;
				}
				if (cmin > j)
				{
					cmin = j;
				}
				if (cmax < j)
				{
					cmax = j;
				}

			}
		}
	}

	par2 = (cmax - cmin + 1.0) / (rmax - rmin + 1.0);

}


Mat_<Vec3b> Projections(const Mat_<Vec3b>& img, const Vec3b& obj_color) {
	int projectionHoriz[1000];
	int projectionVert[1000];
	int temp = 0;
	int i = 0;
	int j = 0;
	int height = img.rows;
	int width = img.cols;
	Mat_<Vec3b> projection_img(img.rows, img.cols, Vec3b(255, 255, 255));

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (img(i, j) == obj_color)
			{
				temp++;
			}
		}

		projectionHoriz[i] = temp;
		temp = 0;
	}

	for (j = 0; j < width; j++)
	{
		for (i = 0; i < height; i++)
		{
			if (img(i, j) == obj_color)
			{
				temp++;
			}
		}

		projectionVert[j] = temp;
		temp = 0;
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < projectionHoriz[i]; j++)
		{
			projection_img(i, j) = obj_color;
		}
	}

	for (j = 0; j < width; j++)
	{
		for (i = 0; i < projectionVert[j]; i++)
		{
			projection_img(i, j) = obj_color;
		}
	}

	return projection_img;
}



void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	int ri;
	int ci;
	int perimeter = 0;
	double angle;
	double par;
	double par2;


	Mat_<Vec3b>img = *((Mat_<Vec3b>*)(param));
	Mat_<Vec3b> perimeter_img(img.rows, img.cols, Vec3b(255, 255, 255));

	Vec3b white = { 255,255,255 };
	if (event == 1)
	{
		Vec3b culoare = img(y, x);
		if (culoare != white) {

			int area = AriaCentruMasa(img, culoare, &ri, &ci);
			printf("Aria : %d\n", area);
			printf("Centrul de masa: %d %d\n", ri, ci);

			AxaAlungire(img, culoare, ri, ci, &angle);
			printf("Unghiul(in radiani): %f\n", angle);


			Perimeter(img, culoare, &perimeter, perimeter_img);
			printf("Perimetrul: %d\n", perimeter);

			FactorSubtiere(area, perimeter, par);
			printf("Factorul de subtiere : %f\n", par);

			FactorAspect(img, culoare, par2);
			printf("Factorul de aspect: %f\n", par2);

			double end_r = ri + 50 * sin(angle);
			double end_r2 = ri + -50 * sin(angle);
			double end_c = ci + 50 * cos(angle);
			double end_c2 = ci + -50 * cos(angle);

			line(perimeter_img, Point(ci, ri), Point(end_c, end_r), Vec3b(0,0,0), 2);
			line(perimeter_img, Point(ci, ri), Point(end_c2, end_r2), Vec3b(0,0,0), 2);
			circle(perimeter_img, Point(ci, ri), 10, Vec3b(0, 0, 0));
			imshow("Punctul b", perimeter_img);
			imshow("Projections", Projections(img, culoare));



		}
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Increase Brightness Additive\n");
		printf(" 11 - Increase Brightness Multiplicative\n");
		printf(" 12 - 4 Colors\n");
		printf(" 13 - Inverse Float Matrix\n");
		printf(" 14 - Split Channels\n");
		printf(" 15 - Convert Grayscale\n");
		printf(" 16 - Convert Grayscale to Black and White\n");
		printf(" 17 - Convert RGB to HSV\n");
		printf(" 18 - Test isInside\n");
		printf(" 19 - Test Histogram\n");
		printf(" 20 - Test FDP\n");
		printf(" 21 - Show Histogram\n");
		printf(" 22 - Reduced Histogram\n");
		printf(" 23 - Multple Steps\n");
		printf(" 24 - Dithering\n");
		printf(" 25 - BFS Labeling\n");
		printf(" 26 - Contur Tracing\n");
		printf(" 27 - Reconstruct\n");
		printf(" 28 - Dilatare\n");
		printf(" 29 - Eroziune\n");
		printf(" 30 - Inchidere\n");
		printf(" 31 - Deschidere\n");
		printf(" 32 - DilatareN\n");
		printf(" 33 - EroziuneN\n");
		printf(" 34 - Extragere Contur\n");
		printf(" 35 - Histograma Cumulativa\n");
		printf(" 36 - Binarizare Automata Globala\n");
		printf(" 37 - Operatii Histograma\n");
		printf(" 38 - Egalizare Histograma\n");
		printf(" 39 - FTJ\n");
		printf(" 40 - FTS\n");
		printf(" 41 - DFT\n");
		printf(" 42 - Median\n");
		printf(" 43 - Gaussian\n");
		printf(" 44 - OptimizedGaussian\n");
		printf(" 45 - Canny\n");
		printf(" 46 - PolygonalAprox\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				incBrightnessAdd();
				break;
			case 11:
				incBrightnessMul();
				break;
			case 12:
				fourColors();
				break;
			case 13:
				inverseFloatMatrix();
				break;
			case 14:
				splitChannels();
				break;
			case 15:
				convertGrayscale();
				break;
			case 16:
				convGrayscaleToBW();
				break;
			case 17:
				convBGR2HSV();
				break;
			case 18:
				testIsInside();
				break;
			case 19:
				testCalculateHistogram();
				break;
			case 20:
				testCalculateFDP();
				break;
			case 21:
				testShowHistogram();
				break;
			case 22:
				testShowReducedHistogram();
				break;
			case 23:
				multipleSteps();
				break;
			case 24:
				dithering();
				break;
			case 25:
				bfs();
				break;
			case 26:
				contur_tracing();
				break;
			case 27:
				reconstruct();
				break;
			case 28:
				dilatare();
				break;
			case 29:
				eroziune();
				break;
			case 30:
				inchidere();
				break;
			case 31:
				deschidere();
				break;
			case 32:
				dilatareN();
				break;
			case 33:
				eroziuneN();
				break;
			case 34:
				extragereContur();
				break;
			case 35:
				calculateHistogramCumulative();
				break;
			case 36:
				binarizareAutomataGlobala();
				break;
			case 37:
				histogramOperations();
				break;
			case 38:
				egalizareHistograma();
				break;
			case 39:
				convolutie(FTJ_INT);
				break;
			case 40:
				convolutie(FTS);
				break;
			case 41:
				dft();
				break;
			case 42:
				testMedianFiltre();
				break;
			case 43:
				testGaussianFiltre();
				break;
			case 44:
				testOptimizedGaussianFiltre();
				break;
			case 45:
				testCannyFunction();
				break;
			case 46:
				polygonal_aprox();
				break;




		}
	}
	while (op!=0);
	return 0;
}