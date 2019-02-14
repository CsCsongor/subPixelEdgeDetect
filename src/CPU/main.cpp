#include "edgeTest.h"
#include "WriteFile.h"
// using namespace std;
using namespace cv;

// Compiling
// g++ main.cpp edgeTest.cpp -w -o test `pkg-config --cflags --libs opencv`


int main()
{
	Mat srcImage, grayImage, dstImage;
	srcImage = imread("images/valve.PNG");
  // srcImage = imread("images/image256.jpeg");
  // srcImage = imread("images/image_512.png");
  // srcImage = imread("images/image_1024.JPG");
  // srcImage = imread("images/image_fhd.jpg");
  // srcImage = imread("images/image_4K.jpg");
	if (srcImage.empty())
	{
		std::cout << "load error!" << std::endl;
		return -1;
	}

	//parameters setting
	double * x;          /* x[n] y[n] coordinates of result contour point n */
	double * y;
	int * curve_limits;  /* limits of the curves in the x[] and y[] */
	int N, M;         /* result: N contour points, forming M curves */
	double S = 1.5; /* default sigma=0 */
	double H = 15; /* default th_h=0  */
	double L = 5; /* default th_l=0  */
	double W = 1.0; /* default W=1.3   */
	char * pdf_out = "output.pdf";  /*pdf filename*/
	char * txt_out = "output.txt";

	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	dstImage = grayImage;
	const int iHeight = dstImage.rows;
	const int iWidth = dstImage.cols;
	uchar* pSrc = grayImage.data;//new uchar[iHeight*iWidth];
	uchar* pDst = dstImage.data;

	std::cout << "******************** Details ************************" << std::endl;
  std::cout << "Image height(rows): " << iHeight << std::endl << "Image width(cols): " << iWidth << std::endl;
	std::cout << "*****************************************************" << std::endl;

	// imshow("input image", grayImage);
	devernay(&x, &y, &N, &curve_limits, &M, pSrc, pDst, iWidth, iHeight, S, H, L);
	// std::cout << "************************************************" << std::endl;
	// std::cout << "Main curve: " << curve_limits << std::endl;
	// std::cout << "Main x: " << x << std::endl;
	// std::cout << "Main y: " << y << std::endl;
	// std::cout << "************************************************" << std::endl;
	// printf("M: %d\n", M);
	if (pdf_out != NULL) write_curves_pdf(x, y, curve_limits, M, pdf_out, iWidth, iHeight, W);
	if (txt_out != NULL) write_curves_txt(x, y, curve_limits, M, txt_out);

	// imshow("gaussion filtered image", dstImage);
	waitKey();
	//system("pause");
	return 0;
}
