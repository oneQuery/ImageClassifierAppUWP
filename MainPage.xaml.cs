// Specify all the using statements which give us the access to all the APIs that you'll need
using System;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Devices.Enumeration;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using WindowsStorageStreams = Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;
using Windows.Storage.Streams;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace ImageClassifierAppUWP
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        // All the required fields declaration
        private ClassifierModel modelGen;
        private ClassifierInput input = new ClassifierInput();
        private ClassifierOutput output;
        private StorageFile selectedStorageFile;
        private string result = "";
        private float resultProbability = 0;

        // The main page to initialize and execute the model.
        public MainPage()
        {
            this.InitializeComponent();
            loadModel();
        }

        private async Task loadModel()
        {
            // Get an access the ONNX model and save it in memory.
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/classifier.onnx"));
            // Instantiate the model. 
            modelGen = await ClassifierModel.CreateFromOnnxModelAsync(modelFile);
        }
        
        // Waiting for a click event to select a file 
        private async void OpenFileButton_Click(object sender, RoutedEventArgs e)
        {
            if (!await getImage())
            {
                return;
            }
            // After the click event happened and an input selected, begin the model execution. 
            // Bind the model input
            await imageBind();
            // Model evaluation
            await evaluate();
            // Extract the results
            extractResult();
            // Display the results  
            await displayResult();
        }

        // A method to select an input image file
        private async Task<bool> getImage()
        {
            try
            {
                // Trigger file picker to select an image file
                FileOpenPicker fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();
                if (selectedStorageFile == null)
                {
                    return false;
                }
            }
            catch (Exception)
            {
                return false;
            }
            return true;
        }

        // A method to convert and bind the input image.  
        private async Task imageBind()
        {
            UIPreviewImage.Source = null;
            try
            {
                SoftwareBitmap softwareBitmap;
                using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                {
                    // Create the decoder from the stream
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                    // Set the desired size
                    BitmapTransform transform = new BitmapTransform
                    {
                        ScaledWidth = 244,
                        ScaledHeight = 244,
                        InterpolationMode = BitmapInterpolationMode.Linear
                    };

                    // Get the SoftwareBitmap representation of the file in BGRA8 format
                    softwareBitmap = await decoder.GetSoftwareBitmapAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied, transform, ExifOrientationMode.RespectExifOrientation, ColorManagementMode.DoNotColorManage);
                }

                // Convert the SoftwareBitmap to a VideoFrame
                VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                // Normalize the input image
                VideoFrame normalizedImage = await NormalizeVideoFrameAsync(inputImage);

                // Bind the input image
                ImageFeatureValue imageTensor = ImageFeatureValue.CreateFromVideoFrame(normalizedImage);
                input.Data = imageTensor;

                // Display the image
                softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
                await imageSource.SetBitmapAsync(softwareBitmap);
                UIPreviewImage.Source = imageSource;
            }
            catch (Exception e)
            {
            }
        }


        private async Task<VideoFrame> NormalizeVideoFrameAsync(VideoFrame inputFrame)
        {
            SoftwareBitmap inputBitmap = inputFrame.SoftwareBitmap;
            int width = inputBitmap.PixelWidth;
            int height = inputBitmap.PixelHeight;

            // Get input pixel data
            byte[] inputPixelData = new byte[width * height * 4]; // Assuming BGRA8 format
            inputBitmap.CopyToBuffer(inputPixelData.AsBuffer());

            // Create normalized pixel data array
            float[] normalizedPixelData = new float[1 * width * height * 3]; // 1 * 3 channels (RGB)

            for (int i = 0; i < width * height; i++)
            {
                int pixelIndex = i * 4;

                // Normalize pixel values to [0, 1] range
                normalizedPixelData[i * 3] = inputPixelData[pixelIndex + 2] / 255.0f; // R
                normalizedPixelData[i * 3 + 1] = inputPixelData[pixelIndex + 1] / 255.0f; // G
                normalizedPixelData[i * 3 + 2] = inputPixelData[pixelIndex] / 255.0f; // B
            }

            // Create a new SoftwareBitmap with the normalized pixel data
            int floatCount = normalizedPixelData.Length;
            byte[] byteArray = new byte[floatCount * sizeof(float)];
            System.Buffer.BlockCopy(normalizedPixelData, 0, byteArray, 0, byteArray.Length);

            // Create a new VideoFrame with a batch dimension
            VideoFrame normalizedVideoFrame = new VideoFrame(BitmapPixelFormat.Rgba16, width, height, 1);

            // Copy the data from the byteArray to the normalizedVideoFrame
            using (var buffer = byteArray.AsBuffer())
            {
                await normalizedVideoFrame.SoftwareBitmap.CopyToBufferAsync(buffer);
            }

            return normalizedVideoFrame;
        }



        // A method to evaluate the model
        private async Task evaluate()
        {
            output = await modelGen.EvaluateAsync(input);
        }

        private void extractResult()
        {
            // A method to extract output (result and a probability) from the "Predictions" output of the model
            var collection = output.Predictions.GetAsVectorView();
            float maxProbability = 0;
            int indexMax = -1;

            for (int i = 0; i < collection.Count; i++)
            {
                if (collection[i] > maxProbability)
                {
                    maxProbability = collection[i];
                    indexMax = i;
                }
            }

            // Assuming you have the list of class labels
            string[] classLabels = new string[] { "falldown", "none" }; // Replace with your actual class labels
            result = classLabels[indexMax];
            resultProbability = maxProbability;
        }

        // A method to display the results
        private async Task displayResult()
        {
            displayOutput.Text = result.ToString();
            displayProbability.Text = resultProbability.ToString();
        }
    }
}
