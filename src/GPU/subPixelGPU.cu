#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <highgui.h>

#include <ctime>

// Compiling
// nvcc $(pkg-config --cflags opencv) $(pkg-config --libs opencv) subPixelGPU.cu -w -o test

using namespace cv;

#define KERNEL_SIZE 7
#define DIM_BLOCK_2D 32

#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)

// Compute a > b considering the rounding errors due to the representation of double numbers
__device__
int greater(double a, double b){
	if (a <= b) return FALSE;
	if ((a - b) < 1000 * DBL_EPSILON) return FALSE;
	return TRUE;
}

// Euclidean distance between x1,y1 and x2,y2
__device__
double dist(double x1, double y1, double x2, double y2){
	return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

__device__
double chain(int from, int to, double * Ex, double * Ey,double * Gx, double * Gy, int rows, int cols){
	double dx, dy;

		// Check that the points are different and valid edge points,otherwise return invalid chaining
		if (from == to){
			// Same pixel, not a valid chaining
			return 0.0;
		}

		if (Ex[from] < 0.0 || Ey[from] < 0.0 || Ex[to] < 0.0 || Ey[to] < 0.0){
			// One of them is not an edge point, not a valid chaining
			return 0.0;
		}

		dx = Ex[to] - Ex[from];
		dy = Ey[to] - Ey[from];
		if ((Gy[from] * dx - Gx[from] * dy) * (Gy[to] * dx - Gx[to] * dy) <= 0.0){
			// Incompatible gradient angles, not a valid chaining
			return 0.0;
		}
		if ((Gy[from] * dx - Gx[from] * dy) >= 0.0){
			return  1.0 /  dist(Ex[from], Ey[from], Ex[to], Ey[to]);	// Forward chaining
		} else {
			return -1.0 /  dist(Ex[from], Ey[from], Ex[to], Ey[to]);	// Backward chaining
		}
}

// Compute a Gaussian kernel of length n, standard deviation sigma,and centered at value mean.
__global__
void cu_gaussian_kernel(double* pfGaussFilter, int n, double sigma, double mean){
	double sum = 0.0;
	double val;
	int x,i;

	for (i=0;i<n;i++){
		val = ((double)i- mean)/ sigma;
		pfGaussFilter[i]= exp(-0.5* val* val);
		sum += pfGaussFilter[i];

	}
	//Normalization
	if (sum > 0.0){
		for (i=0;i<n;i++){
			pfGaussFilter[i]/= sum;
		}
	}
}

/* Filter an image with a Gaussian kernel of parameter sigma. return a pointer
*	 to a newly allocated filtered image, of the same size as the input image.
*/
// X axis convolution
__global__
void cu_gaussian_filterX(uchar* image, int rows, int cols, double* kernel, double* tmp, int offset, int n, double val){
	int x, y, z, i, j, nx2, ny2;

	x = blockIdx.x*blockDim.x+threadIdx.x;
	y = blockIdx.y*blockDim.y+threadIdx.y;

	// Auxiliary variables for the double of the image size
	nx2= 2*rows;

	if ((x < rows) && (y < cols)){
		for (z = 0; z<n; z++){
			j= x- offset+z;

			// Symmetry boundary condition
			while (j<0){
				j+= nx2;
			}
			while (j>=nx2){
				j-= nx2;
			}
			if (j>= rows){
				j=nx2-1-j;
			}
			val+= (double)image[j+y*rows]*kernel[z];
		}
		tmp[x+y*rows]= val;
	}
}

// Y axis convolution
__global__
void cu_gaussian_filterY(uchar* out, int rows, int cols, double* kernel, double* tmp, int offset, int n, double val){
	int x, y, z, i, j, ny2;

	x = blockIdx.x*blockDim.x+threadIdx.x;
	y = blockIdx.y*blockDim.y+threadIdx.y;

	// Auxiliary variables for the double of the image size
	ny2= 2*cols;

	if ((x < rows) && (y < cols)){
		for (z = 0; z<n; z++){
			j= y-offset+z;

			//Symmetry boundary condition
			while (j<0){
				j+=ny2;
			}
			while (j>=ny2){
				j-=ny2;
			}
			if(j>=cols){
				j=ny2-1-j;
			}
			val+= tmp[x+j*rows]*kernel[z];
		}
		out[x+y*rows]= (uchar)val;
	}
}

/* compute the image gradient, giving its x and y components as well as the modulus.
*	 Gx, Gy, and modG must be already allocated.
*/
__global__
void cu_compute_gradient(double * Gx, double * Gy, double * modG, uchar * image, int rows, int cols){
	int x, y;

	x = blockIdx.x*blockDim.x+threadIdx.x+1;
	y = blockIdx.y*blockDim.y+threadIdx.y+1;

	if ((x < (rows-1)) && (y < (cols-1))){
		Gx[x + y*rows] = (double)image[(x + 1) + y*rows] - (double)image[(x - 1) + y*rows];
		Gy[x + y*rows] = (double)image[x + (y + 1)*rows] - (double)image[x + (y - 1)*rows];
		modG[x + y*rows] = sqrt(Gx[x + y*rows] * Gx[x + y*rows] + Gy[x + y*rows] * Gy[x + y*rows]);
	}
}

// Compute sub-pixel edge points using adapted Canny and Devernay methods.
__global__
void cu_compute_edge_points(double * Ex, double * Ey, double * modG, double * Gx, double * Gy, int rows, int cols){
	int x, y;

	x = blockIdx.x*blockDim.x+threadIdx.x+2;
	y = blockIdx.y*blockDim.y+threadIdx.y+2;

	// Explore pixels inside a 2 pixel margin (so modG[x,y +/- 1,1] is defined) */
	if ((x < (rows-2)) && (y < (cols-2))){
		int Dx = 0;                     	 // Interpolation is along Dx,Dy
		int Dy = 0;                     	 // Which will be selected below
		double mod = modG[x + y*rows];     // ModG at pixel
		double L = modG[x - 1 + y*rows];   // ModG at pixel on the left
		double R = modG[x + 1 + y*rows];   // ModG at pixel on the right
		double U = modG[x + (y + 1)*rows]; // ModG at pixel up
		double D = modG[x + (y - 1)*rows]; // ModG at pixel below
		double gx = fabs(Gx[x + y*rows]);  // Absolute value of Gx
		double gy = fabs(Gy[x + y*rows]);  // Absolute value of Gy

		if (greater(mod, L) && !greater(R, mod) && gx >= gy){
			Dx = 1;
		}// Horizontal
		else if (greater(mod, D) && !greater(U, mod) && gx <= gy){
			Dy = 1;
		}// Vertival

		if (Dx > 0 || Dy > 0){
			// Offset value is in [-0.5, 0.5]
			double a = modG[x - Dx + (y - Dy) * rows];
			double b = modG[x + y     * rows];
			double c = modG[x + Dx + (y + Dy) * rows];
			double offset = 0.5 * (a - c) / (a - b - b + c);

			// Store edge point
			Ex[x + y*rows] = x + offset * Dx;
			Ey[x + y*rows] = y + offset * Dy;
		}
	}
}

// Chain edge points
__global__
void cu_chain_edge_points(int * next, int * prev, double * Ex,	double * Ey,double * Gx, double * Gy, int rows, int cols){
	int x, y, i , j, alt;
	int dx, dy, to;

	x = blockIdx.x*blockDim.x+threadIdx.x+2;
	y = blockIdx.y*blockDim.y+threadIdx.y+2;

	// Try each point to make local chains
	// 2 pixel margin to include the tested neighbors
	if (x < (rows-2) && y < (cols-2)){
		// Must be an edge point
		if (Ex[x + y*rows] >= 0.0 && Ey[x + y*rows] >= 0.0){
			int from = x + y*rows;  // Edge point to be chained
			double fwd_s = 0.0;  	  // Score of best forward chaining
			double bck_s = 0.0;     // Score of best backward chaining
			int fwd = -1;           // Edge point of best forward chaining
			int bck = -1;           // Edge point of best backward chaining

			/* try all neighbors two pixels apart or less.
				looking for candidates for chaining two pixels apart, in most such cases,
				is enough to obtain good chains of edge points that	accurately describes the edge.
			*/
			for (i = -2; i <= 2; i++){
				for (j = -2; j <= 2; j++){
					to = x + i + (y + j)*rows; // Candidate edge point to be chained

					double s = chain(from, to, Ex, Ey, Gx, Gy, rows, cols);  //score from-to

					if (s > fwd_s){ // A better forward chaining found
						fwd_s = s;
						fwd = to;
					} else if (s < bck_s){ // A better backward chaining found
						bck_s = s;
						bck = to;
					}
				}
			}

			if (fwd >= 0 && next[from] != fwd && ((alt = prev[fwd]) < 0 || chain(alt, fwd, Ex, Ey, Gx, Gy, rows, cols) < fwd_s)){
				if (next[from] >= 0){			// Remove previous from-x link if one */
					prev[next[from]] = -1;	// Only prev requires explicit reset  */
				}
				next[from] = fwd;					// Set next of from-fwd link          */
				if (alt >= 0){						// Remove alt-fwd link if one
					next[alt] = -1;					// Only next requires explicit reset
				}
				prev[fwd] = from;					// Set prev of from-fwd link
			}
			if (bck >= 0 && prev[from] != bck && ((alt = next[bck]) < 0 || chain(alt, bck, Ex, Ey, Gx, Gy, rows, cols) > bck_s)){
					if (alt >= 0){					// Remove bck-alt link if one
						prev[alt] = -1;				// Only prev requires explicit reset
					}
					next[bck] = from;				// Set next of bck-from link
					if (prev[from] >= 0){		// Remove previous x-from link if one
						next[prev[from]] = -1; // Only next requires explicit reset
					}
					prev[from] = bck;				// Set prev of bck-from link
			}
		}
	}
}

// Apply Canny thresholding with hysteresis
__global__
void cu_thresholds_with_hysteresis(int * next, int * prev, double * modG,	int rows, int cols, double th_h, double th_l, int * valid){
	int x, y, j, k;

	x = blockIdx.x*blockDim.x+threadIdx.x;
	y = blockIdx.y*blockDim.y+threadIdx.y;
	int idx = x+y*rows;

	// Validate all edge points over th_h or connected to them and over th_l
	if (idx < rows*cols){
		if ((prev[idx] >= 0 || next[idx] >= 0) && !valid[idx] && modG[idx] >= th_h){
			valid[idx] = TRUE; // Mark as valid the new point

			// Follow the chain of edge points forwards
			for (j = idx; j >= 0 && (k = next[j]) >= 0 && !valid[k]; j = next[j]){
				if (modG[k] < th_l){
					next[j] = -1;	// Cut the chain when the point is below th_l
					prev[k] = -1;
				} else {
					valid[k] = TRUE;
				}
			}
			for (j = idx; j >= 0 && (k = prev[j]) >= 0 && !valid[k]; j = prev[j]){
				if (modG[k] < th_l){
					prev[j] = -1;	// Cut the chain when the point is below th_l
					next[k] = -1;
				} else {
					valid[k] = TRUE;
				}
			}
		}
	}
}

// Remove any remaining non-valid chained point
__global__
void cu_thresholds_remove(int * next, int * prev,int rows, int cols, int * valid){
	int x, y;

	x = blockIdx.x*blockDim.x+threadIdx.x;
	y = blockIdx.y*blockDim.y+threadIdx.y;
	int idx = x+y*rows;

	if (idx < rows*cols){
		if ((prev[idx] >= 0 || next[idx] >= 0) && !valid[idx]){
			prev[idx] = next[idx] = -1;
		}
	}
}

/* create a list of chained edge points composed of 3 lists
*  x, y and curve_limits; it also computes N (the number of edge points) and
*	 M (the number of curves).
*/
void list_chained_edge_points(double ** x, double ** y, int * N,int ** curve_limits,
	int * M,int * next, int * prev,	double * Ex, double * Ey, int rows, int cols)
{
	int k, n, m, l, i;

	// Initialize output: x, y, curve_limits, N, and M
	*x = (double*)malloc(rows*cols*sizeof(double));
	*y = (double*)malloc(rows*cols*sizeof(double));
	*curve_limits = (int *)malloc(rows*cols*sizeof(int));
	*N= 0;
	*M= 0;

	// Copy chained edge points to output
	for (i=0; i<rows*cols; i++){
		if (prev[i] >= 0 || next[i] >= 0){
			(*curve_limits)[*M] = *N;
			++(*M);

			// Set k to the beginning of the chain, or to i if closed curve
			// for (k = i; (n = prev[k]) >= 0 && n != i; k = n);
			if ((n = prev[i]) >=0 && n != i){ k= n;}

			do{
				(*x)[*N] = Ex[k];
				(*y)[*N] = Ey[k];
				++(*N);

				// Save the id of the next point in the chain
				n = next[k];

				// Unlink chains from k so it is not used again
				next[k] = -1;
				prev[k] = -1;

				// Set the current point to the next in the chain
				k = n;
			} while (k >= 0); // Continue while there is a next point in the chain
		}
		// Store end of the last chain
		(*curve_limits)[*M] = *N;
	}
}


/* Entry point for serial program calling CUDA implementation
*	 Chained, sub-pixel edge detector. Based on a modified Canny non-maximal
*  suppression and a modified Devernay sub-pixel correction.
*/
void devernay(double ** x, double ** y, int * N, int ** curve_limits,int * M,
	uchar * image, uchar * gauss, int rows, int cols, double sigma, double th_high, double th_low){

	//Gaussian kernel
	double prec;
	int offset, n;
	prec = 3.0;
	offset = (int)ceil(sigma* sqrt(2.0* prec* log(10.0)));
	n = 1+ 2* offset; // Kernel size
	double val = 0.0;

	dim3 numberOfBlocks((rows+DIM_BLOCK_2D-1)/DIM_BLOCK_2D, (cols+ DIM_BLOCK_2D-1)/DIM_BLOCK_2D); //, n);
	dim3 threadsPerBlock(DIM_BLOCK_2D, DIM_BLOCK_2D);;

	// Allocate device memory
  double * Gx;
  cudaMalloc(&Gx, (rows*cols*sizeof(double)));      // grad_x
  double * Gy;
  cudaMalloc(&Gy, (rows*cols*sizeof(double)));      // grad_y
  double * modG;
  cudaMalloc(&modG, (rows*cols*sizeof(double)));  	// |grad|
  double * Ex;
  cudaMalloc(&Ex, (rows*cols*sizeof(double)));      // edge_x
  double * Ey;
  cudaMalloc(&Ey, (rows*cols*sizeof(double)));      // edge_y
  int * next;
  cudaMalloc(&next, (rows*cols*sizeof(int)));     	// next point in chain
  int * prev;
  cudaMalloc(&prev, (rows*cols*sizeof(int)));     	// prev point in chain
	//
	uchar * img;
	cudaMalloc(&img, (rows*cols*sizeof(uchar)));
	double * kernel;
	cudaMalloc(&kernel, (rows*cols*sizeof(double)));
	double * tmp;
	cudaMalloc(&tmp, (rows*cols*sizeof(double)));
	uchar * gaussK;
	cudaMalloc(&gaussK, (rows*cols*sizeof(uchar)));
	// cu_compute_edge_points
	double * tmpEx;
  tmpEx = (double*)malloc(rows*cols*sizeof(double));
  double * tmpEy;
  tmpEy = (double*)malloc(rows*cols*sizeof(double));
	// Initialize Ex and Ey as non-edge points for all pixels
	for (int i = 0; i<rows*cols; i++){
		tmpEx[i] = tmpEy[i] = -1.0;
	}
	cudaMemcpy(Ex, tmpEx, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Ey, tmpEy, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
	// cu_chain_edge_points
	int * tmpNext;
	tmpNext = (int*)malloc(rows*cols*sizeof(int));
	int * tmpPrev;
	tmpPrev = (int*)malloc(rows*cols*sizeof(int));
	// Initialize next and prev as non linked
	for (int i = 0; i<rows*cols; i++){
		tmpNext[i] = tmpPrev[i] = -1;
	}
	cudaMemcpy(next, tmpNext, rows*cols*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(prev, tmpPrev, rows*cols*sizeof(int), cudaMemcpyHostToDevice);
	//cu_thresholds_with_hysteresis
	int * valid;
	cudaMalloc(&valid, (rows*cols*sizeof(int)));
	int * tmpValid;
	tmpValid = (int*)malloc(rows*cols*sizeof(int));
	for (int i = 0; i<rows*cols; i++){
		tmpValid[i] = FALSE;
	}
	cudaMemcpy(valid, tmpValid, rows*cols*sizeof(int), cudaMemcpyHostToDevice);

	// Data transfer image pixel to device
	cudaMemcpy(img, image, rows*cols*sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(gaussK, gauss, rows*cols*sizeof(uchar), cudaMemcpyHostToDevice);

	// Run Sub-Pixel edge detection core - CUDA kernels
  // Use streams to ensure the kernels are in the same task
  cudaStream_t stream;
  cudaStreamCreate(&stream);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	if (sigma == 0.0){
		cu_compute_gradient<<<numberOfBlocks, threadsPerBlock, 0, stream>>>(Gx, Gy, modG, image, rows, cols);
	} else {
		cu_gaussian_kernel<<<1, 1>>>(kernel, n, sigma, (double)offset);
		cudaDeviceSynchronize();
		cu_gaussian_filterX<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(img, rows, cols, kernel, tmp, offset, n, val);
		cudaDeviceSynchronize();
		cu_gaussian_filterY<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(gaussK, rows, cols, kernel, tmp, offset, n, val);
		cudaDeviceSynchronize();
		cu_compute_gradient<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(Gx, Gy, modG, gaussK, rows, cols);
	}
	cudaDeviceSynchronize();
	cu_compute_edge_points<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(Ex, Ey, modG, Gx, Gy, rows, cols);
	cudaDeviceSynchronize();
	cu_chain_edge_points<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(next, prev, Ex, Ey, Gx, Gy, rows, cols);
	cudaDeviceSynchronize();
	cu_thresholds_with_hysteresis<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(next, prev, modG, rows, cols, th_high, th_low, valid);
  cudaDeviceSynchronize();
	cu_thresholds_remove<<<numberOfBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(uchar), stream>>>(next, prev, rows, cols, valid);
	cudaDeviceSynchronize();

	cudaMemcpy(tmpEx, Ex, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpEy, Ey, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpNext, next, rows*cols*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpPrev, prev, rows*cols*sizeof(int), cudaMemcpyDeviceToHost);
	list_chained_edge_points(x, y, N, curve_limits, M, tmpNext, tmpPrev, tmpEx, tmpEy, rows, cols);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float duration = 0;
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << "Elapsed time: " << duration << "ms\n";

	// Cleanup
  cudaFree(Gx);
  cudaFree(Gy);
  cudaFree(modG);
  cudaFree(Ex);
  cudaFree(Ey);
  cudaFree(next);
  cudaFree(prev);
  cudaFree(img);
  cudaFree(kernel);
  cudaFree(tmp);
  cudaFree(gaussK);
  cudaFree(valid);
}


FILE * xfopen(const char * path, const char * mode){
	FILE * f = fopen(path, mode);
	return f;
}

int xfclose(FILE * f){
	if (fclose(f) == EOF);
	return 0;
}

void write_curves_pdf(double * x, double * y, int * curve_limits, int M, char * filename, int X, int Y, double width){
	FILE * pdf;
	long start1, start2, start3, start4, start5, startxref, stream_len;
	int i, j, k;

	if (M > 0 && (x == NULL || y == NULL || curve_limits == NULL))
		std::cout << "Invalid curves data in write_curves_pdf" << std::endl;
	if (X <= 0 || Y <= 0)
		std::cout << "Invalid image size in write_curves_pdf" << std::endl;

	pdf = xfopen(filename, "wb");

	fprintf(pdf, "%%PDF-1.4\n");
	fprintf(pdf, "%%%c%c%c%c\n", 0xe2, 0xe3, 0xcf, 0xd3);

	start1 = ftell(pdf);
	fprintf(pdf, "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\n");
	fprintf(pdf, "endobj\n");
	start2 = ftell(pdf);
	fprintf(pdf, "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1 ");
	fprintf(pdf, "/Resources <<>> /MediaBox [0 0 %d %d]>>\nendobj\n", X, Y);
	start3 = ftell(pdf);
	fprintf(pdf, "3 0 obj\n");
	fprintf(pdf, "<</Type /Page /Parent 2 0 R /Contents 4 0 R>>\n");
	fprintf(pdf, "endobj\n");

	start4 = ftell(pdf);
	fprintf(pdf, "4 0 obj\n<</Length 5 0 R>>\n");
	fprintf(pdf, "stream\n");
	stream_len = ftell(pdf);
	fprintf(pdf, "%.4f w\n", width);
	for (k = 0; k<M; k++)	{
		i = curve_limits[k];
		fprintf(pdf, "%.4f %.4f m\n", x[i] + 0.5, Y - y[i] - 0.5);

		for (j = i + 1; j<curve_limits[k + 1]; j++)
			fprintf(pdf, "%.4f %.4f l\n", x[j] + 0.5, Y - y[j] - 0.5);

		j = curve_limits[k + 1] - 1;
		if (x[i] == x[j] && y[i] == y[j]) fprintf(pdf, "h\n");

		fprintf(pdf, "S\n");
	}
	stream_len = ftell(pdf) - stream_len;
	fprintf(pdf, "\r\nendstream\n");
	fprintf(pdf, "endobj\n");

	start5 = ftell(pdf);
	fprintf(pdf, "5 0 obj\n%ld\nendobj\n", stream_len);

	startxref = ftell(pdf);
	fprintf(pdf, "xref\n");
	fprintf(pdf, "0 6\n");
	fprintf(pdf, "0000000000 65535 f\r\n");
	fprintf(pdf, "%010ld 00000 n\r\n", start1);
	fprintf(pdf, "%010ld 00000 n\r\n", start2);
	fprintf(pdf, "%010ld 00000 n\r\n", start3);
	fprintf(pdf, "%010ld 00000 n\r\n", start4);
	fprintf(pdf, "%010ld 00000 n\r\n", start5);

	fprintf(pdf, "trailer <</Size 6 /Root 1 0 R>>\n");
	fprintf(pdf, "startxref\n");
	fprintf(pdf, "%ld\n", startxref);
	fprintf(pdf, "%%%%EOF\n");

	xfclose(pdf);
}

int main(int argc, char* argv[]){
  Mat srcImage, grayImage, dstImage;

  // Reading the image
  srcImage = imread("images/valve.PNG");
  // srcImage = imread("images/image256.jpeg");
  // srcImage = imread("images/image_512.png");
  // srcImage = imread("images/image_1024.JPG");
  // srcImage = imread("images/image_fhd.jpg");
  // srcImage = imread("images/image_4K.jpg");
  if (srcImage.empty()){
    std::cout << "Image load error!" << std::endl;
    return -1;
  }

  // Parameters setting
  double * x;                      /* x[n] y[n] coordinates of result contour point n */
	double * y;
	int * curve_limits;              /* limits of the curves in the x[] and y[] */
	int N, M;                        /* result: N contour points, forming M curves */
	double S = 1.5;                  /* default sigma=0 */
	double H = 15;//4.2;             /* default th_h=0  */ //5
	double L = 5;// 0.81;            /* default th_l=0  */ //7
	double W = 1.0;                  /* default W=1.3   */
	char * pdf_out = "output.pdf";   /*pdf filename*/
	char * txt_out = "output.txt";

  cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
  dstImage = grayImage;
  const int imgHeight = dstImage.rows;
  const int imgWidth = dstImage.cols;
  uchar* pSrc = grayImage.data;
  uchar* pDst = dstImage.data;

	std::cout << "******************** Details ************************" << std::endl;
  std::cout << "Image height(rows): " << imgHeight << std::endl << "Image width(cols): " << imgWidth << std::endl;
	std::cout << "*****************************************************" << std::endl;

	devernay(&x, &y, &N, &curve_limits, &M, pSrc, pDst, imgWidth, imgHeight, S, H, L);
	std::cout << "*****************************************************" << std::endl;

	if (pdf_out != NULL){
		write_curves_pdf(x, y, curve_limits, M, pdf_out, imgWidth, imgHeight, W);
	}

  return 0;
}
