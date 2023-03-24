#include <opencv2/opencv.hpp>
#include "tensorflow/lite/c/c_api.h"

using namespace std;
using namespace cv;
int main()
{

	Mat img = imread("1.jpg");
	Mat* input = &img;
	TfLiteModel* model = TfLiteModelCreateFromFile("mnist.tflite");
	TfLiteInterpreterOptions* option = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(option, 0);
	TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, option);
	TfLiteInterpreterAllocateTensors(interpreter);
	TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
	TfLiteTensorCopyFromBuffer(input_tensor, input, img.cols * img.rows * img.channels() * sizeof(float));
	TfLiteInterpreterInvoke(interpreter);
	const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
	float logits[10];
	TfLiteTensorCopyToBuffer(output_tensor, logits, 10 * sizeof(float));

	float maxV = -1;
	int maxIdx = -1;
	for (int i = 0; i < 10; i++)
	{
		if (logits[i] > maxV)
		{
			maxV = logits[i];
			maxIdx = i;
		}
		printf("%d->%f\n", i, logits[i]);
	}
	cout << "Àà±ð£º" << maxIdx << "£¬¸ÅÂÊ£º" << maxV << endl;
	TfLiteInterpreterDelete(interpreter);
	TfLiteInterpreterOptionsDelete(option);
	TfLiteModelDelete(model);
	cout << "-----------" << endl;
	return 0;
}
