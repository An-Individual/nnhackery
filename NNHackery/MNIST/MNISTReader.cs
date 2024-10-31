using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.MNIST
{
    /// <summary>
    /// Data comes from https://github.com/cvdfoundation/mnist.
    /// </summary>
    public static class MNISTReader
    {
        private const string TEST_IMAGES_FILE = "t10k-images.idx3-ubyte";
        private const string TEST_LABELS_FILE = "t10k-labels.idx1-ubyte";
        private const string TRAINING_IMAGES_FILE = "train-images.idx3-ubyte";
        private const string TRAINING_LABELS_FILE = "train-labels.idx1-ubyte";

        private static Stream OpenEmbeddedFile(string file)
        {
            return Assembly.GetExecutingAssembly().GetManifestResourceStream($"NNHackery.MNIST.Data.{file}") ?? throw new Exception("Resource not found");
        }

        public static MNISTTestImage[] ReadTestImages()
        {
            return ReadImagesFromStream(OpenEmbeddedFile(TEST_IMAGES_FILE), OpenEmbeddedFile(TEST_LABELS_FILE));
        }

        public static MNISTTestImage[] ReadTrainingImages()
        {
            return ReadImagesFromStream(OpenEmbeddedFile(TRAINING_IMAGES_FILE), OpenEmbeddedFile(TRAINING_LABELS_FILE));
        }

        private static MNISTTestImage[] ReadImagesFromStream(Stream imageStream, Stream labelStream)
        {
            IDXHeader imagesHeader = IDXHeader.ReadHeader(imageStream);
            IDXHeader labelsHeader = IDXHeader.ReadHeader(labelStream);

            if(imagesHeader.Dimensions.Count != 3)
            {
                throw new Exception("Unexpected number of image headers.");
            }

            if (labelsHeader.Dimensions.Count != 1)
            {
                throw new Exception("Unexpected number of label headers.");
            }

            int imageCount = imagesHeader.Dimensions[0];
            int labelCount = labelsHeader.Dimensions[0];

            if(imageCount != labelCount)
            {
                throw new Exception("Image and label counts do not match.");
            }

            int imageWidth = imagesHeader.Dimensions[1];
            int imageHeight = imagesHeader.Dimensions[2];

            MNISTTestImage[] result = new MNISTTestImage[imageCount];

            for (int i = 0; i < imageCount; i++)
            {
                byte label = (byte)labelStream.ReadByte();

                byte[] imageRaw = new byte[imageWidth * imageHeight];
                imageStream.ReadExactly(imageRaw);
                double[] image = imageRaw.Select(b => b / 255.0).ToArray();

                result[i] = new MNISTTestImage()
                {
                    Label = label,
                    LabelVector = MakeLabelVector(label),
                    FlattenedImage = Vector<double>.Build.Dense(image),
                    Width = imageWidth,
                    Height = imageHeight
                };
            }

            return result;
        }

        private static Vector<double> MakeLabelVector(byte label)
        {
            if(label >= 10)
            {
                throw new Exception("Label is not a single digit.");
            }

            Vector<double> result = Vector<double>.Build.Dense(10);
            result[label] = 1;
            return result;
        }
    }
}
